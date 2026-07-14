const webUrl = process.env.TESTMU_WEB_URL;
const desktopName = process.env.TESTMU_DEVICE || "Linux Desktop";
const platformName = process.env.TESTMU_PLATFORM_NAME || "Linux";
const browserName = process.env.TESTMU_BROWSER_NAME || "Chrome";
const browserVersion = process.env.TESTMU_BROWSER_VERSION || "latest";
const tunnelName = process.env.TESTMU_TUNNEL_NAME;

if (!webUrl) {
  throw new Error(
    "TESTMU_WEB_URL must be set to the locally hosted benchmark URL."
  );
}

if (!tunnelName) {
  throw new Error("TESTMU_TUNNEL_NAME must be set for TestMu desktop web runs.");
}

exports.config = {
  user: process.env.LT_USERNAME,
  key: process.env.LT_ACCESS_KEY,

  protocol: "https",
  hostname: "hub.lambdatest.com",
  port: 443,
  path: "/wd/hub",

  specs: ["./specs/kittentts-benchmark.mobile-web.spec.js"],
  exclude: [],
  maxInstances: 1,

  capabilities: [
    {
      browserName,
      browserVersion,
      "LT:Options": {
        platformName,
        build: process.env.GITHUB_RUN_ID
          ? `KittenTTS RN web ${process.env.GITHUB_RUN_ID}`
          : "KittenTTS RN web local",
        name: `KittenTTS benchmark web ${browserName} - ${desktopName}`,
        project: "KittenTTS React Native",
        tunnel: true,
        tunnelName,
        resolution: process.env.TESTMU_DESKTOP_RESOLUTION || "1920x1080",
        selenium_version: "4.0.0",
        w3c: true,
        console: true,
        visual: true,
        network: false,
      },
    },
  ],

  logLevel: "info",
  bail: 0,
  waitforTimeout: 60000,
  connectionRetryTimeout: Number(
    process.env.TESTMU_CONNECTION_RETRY_TIMEOUT_MS || 900000
  ),
  connectionRetryCount: Number(process.env.TESTMU_CONNECTION_RETRY_COUNT || 1),

  framework: "mocha",
  reporters: ["spec"],
  mochaOpts: {
    ui: "bdd",
    timeout: 1800000,
  },
};
