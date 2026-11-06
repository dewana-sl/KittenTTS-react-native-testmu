// One TestMu (LambdaTest) real device running the Kitten TTS 2 benchmark app,
// with the tunnel the workflow opened to the runner serving the model files.
const platformName = process.env.TESTMU_PLATFORM || 'iOS';
const deviceName = process.env.TESTMU_DEVICE;
const platformVersion = process.env.TESTMU_VERSION;
const tunnelName = process.env.TESTMU_TUNNEL_NAME;

for (const [name, value] of Object.entries({ TESTMU_APP_URL: process.env.TESTMU_APP_URL, TESTMU_DEVICE: deviceName, TESTMU_VERSION: platformVersion, TESTMU_TUNNEL_NAME: tunnelName })) {
  if (!value) throw new Error(`${name} must be set.`);
}

exports.config = {
  user: process.env.LT_USERNAME,
  key: process.env.LT_ACCESS_KEY,
  protocol: 'https',
  hostname: 'mobile-hub.lambdatest.com',
  port: 443,
  path: '/wd/hub',

  specs: ['./tts2.spec.js'],
  maxInstances: 1,
  capabilities: [{
    platformName,
    'appium:deviceName': deviceName,
    'appium:platformVersion': platformVersion,
    'appium:app': process.env.TESTMU_APP_URL,
    'appium:autoAcceptAlerts': platformName === 'iOS',
    'appium:autoGrantPermissions': platformName === 'Android',
    'appium:newCommandTimeout': 900,
    'lt:options': {
      build: `Kitten TTS 2 on-device ${process.env.GITHUB_RUN_ID || 'local'}`,
      name: `Kitten TTS 2 ${platformName} - ${deviceName}`,
      project: 'KittenTTS React Native',
      isRealMobile: true,
      tunnel: true,
      tunnelName,
      idleTimeout: 1800,
      console: true,
      devicelog: true,
      network: false,
      visual: false,
    },
  }],

  logLevel: 'warn',
  bail: 0,
  waitforTimeout: 30000,
  connectionRetryTimeout: 300000,
  connectionRetryCount: 2,
  framework: 'mocha',
  reporters: ['spec'],
  mochaOpts: { ui: 'bdd', timeout: 60 * 60 * 1000 },
};
