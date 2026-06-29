const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

module.exports = mergeConfig(getDefaultConfig(__dirname), {
  resolver: {
    resolverMainFields: ['react-native', 'main'],
  },
});
