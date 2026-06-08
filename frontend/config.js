/**
 * Frontend integration settings.
 * Override wsBaseUrl per environment (dev/staging/prod).
 */
window.VOICE_AGENT_CONFIG = {
  wsBaseUrl: "ws://localhost:8854/ws",
  defaultLang: "en-IN",
  defaultVoice: "netra",
  voices: [
    { id: "netra", label: "Netra" },
    { id: "parvaty", label: "Parvaty" },
    { id: "rohan", label: "Rohan" },
    { id: "sneha", label: "Sneha" },
    { id: "vishal", label: "Vishal" },
    { id: "sagar", label: "Sagar" },
  ],
};
