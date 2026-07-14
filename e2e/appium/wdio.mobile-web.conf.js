const webUrl = process.env.TESTMU_WEB_URL;
const deviceName = process.env.TESTMU_DEVICE || "Pixel 8";
const platformName = process.env.TESTMU_PLATFORM_NAME || "Android";
const platformVersion = process.env.TESTMU_PLATFORM_VERSION || "14";
const browserName = process.env.TESTMU_BROWSER_NAME || "Chrome";
const tunnelName = process.env.TESTMU_TUNNEL_NAME;

if (!webUrl) {
  throw new Error(
    "TESTMU_WEB_URL must be set to the locally hosted benchmark URL."
  );
}

if (!tunnelName) {
  throw new Error("TESTMU_TUNNEL_NAME must be set for TestMu mobile web runs.");
}

exports.config = {
  user: process.env.LT_USERNAME,
  key: process.env.LT_ACCESS_KEY,

  protocol: "https",
  hostname: "mobile-hub.lambdatest.com",
  port: 443,
  path: "/wd/hub",

  specs: ["./specs/kittentts-benchmark.mobile-web.spec.js"],
  exclude: [],
  maxInstances: 1,

  capabilities: [
    {
      platformName,
      browserName,
      "appium:deviceName": deviceName,
      "appium:platformVersion": platformVersion,
      "appium:newCommandTimeout": 900,
      "lt:options": {
        build: process.env.GITHUB_RUN_ID
          ? `KittenTTS RN web ${process.env.GITHUB_RUN_ID}`
          : "KittenTTS RN web local",
        name: `KittenTTS benchmark web ${browserName} - ${deviceName}`,
        project: "KittenTTS React Native",
        isRealMobile: process.env.TESTMU_REAL_DEVICE !== "false",
        tunnel: true,
        tunnelName,
        console: true,
        visual: true,
        network: false,
        devicelog: true,
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
