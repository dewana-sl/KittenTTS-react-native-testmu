const appUrl = process.env.TESTMU_IOS_APP_URL;
const deviceName = process.env.TESTMU_IOS_DEVICE || "iPhone 14";
const platformVersion = process.env.TESTMU_IOS_VERSION || "16";

if (!appUrl) {
  throw new Error(
    "TESTMU_IOS_APP_URL must be set to a lt:// iOS app URL returned by TestMu."
  );
}

exports.config = {
  user: process.env.LT_USERNAME,
  key: process.env.LT_ACCESS_KEY,

  hostname: "mobile-hub.lambdatest.com",
  port: 80,
  path: "/wd/hub",

  specs: ["./specs/kittentts-benchmark.android.spec.js"],
  exclude: [],
  maxInstances: 1,

  capabilities: [
    {
      platformName: "iOS",
      "appium:deviceName": deviceName,
      "appium:platformVersion": platformVersion,
      "appium:app": appUrl,
      "appium:autoAcceptAlerts": true,
      "appium:newCommandTimeout": 900,
      "lt:options": {
        build: process.env.GITHUB_RUN_ID
          ? `KittenTTS RN ${process.env.GITHUB_RUN_ID}`
          : "KittenTTS RN local",
        name: `KittenTTS benchmark app iOS - ${deviceName}`,
        project: "KittenTTS React Native",
        isRealMobile: process.env.TESTMU_REAL_DEVICE !== "false",
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
  connectionRetryTimeout: 300000,
  connectionRetryCount: 2,

  framework: "mocha",
  reporters: ["spec"],
  mochaOpts: {
    ui: "bdd",
    timeout: 1800000,
  },
};
