#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Ministros Voice Agent — Azure Deployment Script
# Deploys backend to Azure Container Apps + frontend to Static Web Apps
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker installed and running
#   - $1000 Azure startup credits active
#
# Usage:
#   chmod +x azure-deploy.sh
#   ./azure-deploy.sh
# ──────────────────────────────────────────────────────────

set -euo pipefail

# ─── Configuration ────────────────────────────────────────
RESOURCE_GROUP="ministros-voice-rg"
LOCATION="centralindia"                # Closest to Sarvam AI servers
ACR_NAME="ministrosacr"                # Azure Container Registry name (must be globally unique)
CONTAINER_ENV="ministros-env"
CONTAINER_APP="ministros-voice-agent"
IMAGE_NAME="ministros-voice-agent"
IMAGE_TAG="latest"

echo "═══════════════════════════════════════════════════"
echo "  Ministros Voice Agent — Azure Deployment"
echo "═══════════════════════════════════════════════════"

# ─── Step 1: Create Resource Group ────────────────────────
echo ""
echo "▸ Creating resource group: $RESOURCE_GROUP in $LOCATION..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ─── Step 2: Create Container Registry ────────────────────
echo "▸ Creating Azure Container Registry: $ACR_NAME..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none

# ─── Step 3: Build & Push Docker Image ────────────────────
echo "▸ Building and pushing Docker image..."
az acr build \
    --registry "$ACR_NAME" \
    --image "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file Dockerfile \
    .

# ─── Step 4: Create Container Apps Environment ────────────
echo "▸ Creating Container Apps environment: $CONTAINER_ENV..."
az containerapp env create \
    --name "$CONTAINER_ENV" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ─── Step 5: Get ACR Credentials ─────────────────────────
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# ─── Step 6: Deploy Container App ─────────────────────────
echo "▸ Deploying container app: $CONTAINER_APP..."
echo ""
echo "  ⚠  You will need to set the following secrets after deployment:"
echo "     SARVAM_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY"
echo ""

az containerapp create \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_ENV" \
    --image "${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_NAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8805 \
    --ingress external \
    --transport auto \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --env-vars \
        "HOST=0.0.0.0" \
        "PORT=8805" \
        "CORS_ORIGINS=*" \
    --output none

# ─── Step 7: Set Secrets ──────────────────────────────────
echo ""
echo "▸ Setting API key secrets..."
echo "  Enter your API keys when prompted (they will be stored securely):"
echo ""

read -sp "  SARVAM_API_KEY: " SARVAM_KEY && echo ""
read -sp "  CEREBRAS_API_KEY: " CEREBRAS_KEY && echo ""
read -sp "  GROQ_API_KEY: " GROQ_KEY && echo ""

az containerapp secret set \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --secrets \
        "sarvam-api-key=$SARVAM_KEY" \
        "cerebras-api-key=$CEREBRAS_KEY" \
        "groq-api-key=$GROQ_KEY" \
    --output none

# Update container to use secrets as env vars
az containerapp update \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --set-env-vars \
        "SARVAM_API_KEY=secretref:sarvam-api-key" \
        "CEREBRAS_API_KEY=secretref:cerebras-api-key" \
        "GROQ_API_KEY=secretref:groq-api-key" \
    --output none

# ─── Step 8: Get Deployment URL ───────────────────────────
APP_URL=$(az containerapp show \
    --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" \
    -o tsv)

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Deployment Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Backend URL:    https://${APP_URL}"
echo "  Health Check:   https://${APP_URL}/health"
echo "  Ready Check:    https://${APP_URL}/ready"
echo "  WebSocket:      wss://${APP_URL}/ws"
echo ""
echo "  Next steps:"
echo "  1. Update your frontend .env.local:"
echo "     NEXT_PUBLIC_WS_URL=wss://${APP_URL}/ws"
echo ""
echo "  2. Deploy frontend to Azure Static Web Apps:"
echo "     az staticwebapp create \\"
echo "       --name ministros-frontend \\"
echo "       --resource-group $RESOURCE_GROUP \\"
echo "       --source https://github.com/ForcedScripter/NaturalVoclAI \\"
echo "       --branch main \\"
echo "       --app-location / \\"
echo "       --output-location .next"
echo ""
echo "  3. Update CORS_ORIGINS to restrict to your frontend domain"
echo ""
echo "  Estimated monthly cost: ~\$50 (well within \$1000 credits)"
echo "═══════════════════════════════════════════════════"
