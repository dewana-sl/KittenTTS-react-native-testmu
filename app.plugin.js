const fs = require('fs');
const path = require('path');
const PbxFile = require('xcode/lib/pbxFile');
const {
  IOSConfig,
  withDangerousMod,
  withGradleProperties,
  withPodfileProperties,
  withXcodeProject,
} = require('expo/config-plugins');

const DEFAULT_ASSETS_DIR = './assets/kittentts';
const DEFAULT_ASSET_ROOT = 'kittentts';
// MLX, which runs Kitten TTS 2's language model on the GPU, needs iOS 17.
const MLX_IOS_DEPLOYMENT_TARGET = '17.0';

module.exports = function withKittenTTS(config, props = {}) {
  const assetsDir = props.assetsDir || DEFAULT_ASSETS_DIR;
  const assetRoot = props.assetRoot || DEFAULT_ASSET_ROOT;

  if (props.mlx) {
    // KittenTTSReactNative.podspec reads this and builds the MLX module.
    config = withPodfileProperties(config, (nextConfig) => {
      nextConfig.modResults['kittentts.mlx'] = 'true';
      const current = nextConfig.modResults['ios.deploymentTarget'];
      if (!current || parseFloat(current) < parseFloat(MLX_IOS_DEPLOYMENT_TARGET)) {
        nextConfig.modResults['ios.deploymentTarget'] = MLX_IOS_DEPLOYMENT_TARGET;
      }
      return nextConfig;
    });
  }

  if (props.ggml) {
    // android/build.gradle reads this and builds the ggml engine (Vulkan on the GPU).
    config = withGradleProperties(config, (nextConfig) => {
      const properties = nextConfig.modResults.filter(
        (item) => !(item.type === 'property' && item.key === 'kittentts.ggml'),
      );
      properties.push({ type: 'property', key: 'kittentts.ggml', value: 'true' });
      nextConfig.modResults = properties;
      return nextConfig;
    });
  }

  config = withDangerousMod(config, [
    'android',
    async (nextConfig) => {
      const source = path.resolve(nextConfig.modRequest.projectRoot, assetsDir);
      const destination = path.join(
        nextConfig.modRequest.platformProjectRoot,
        'app',
        'src',
        'main',
        'assets',
        assetRoot,
      );
      copyDirectory(source, destination);
      return nextConfig;
    },
  ]);

  config = withXcodeProject(config, async (nextConfig) => {
    const source = path.resolve(nextConfig.modRequest.projectRoot, assetsDir);
    const iosSourceRoot = IOSConfig.Paths.getSourceRoot(nextConfig.modRequest.projectRoot);
    const destination = path.join(iosSourceRoot, assetRoot);
    copyDirectory(source, destination);

    addFolderReferenceToResources(
      nextConfig.modResults,
      path.relative(nextConfig.modRequest.platformProjectRoot, destination),
    );

    return nextConfig;
  });

  return config;
};

function copyDirectory(source, destination) {
  if (!fs.existsSync(source)) return;
  fs.rmSync(destination, { recursive: true, force: true });
  fs.mkdirSync(destination, { recursive: true });
  fs.cpSync(source, destination, { recursive: true });
}

function addFolderReferenceToResources(project, folderPath) {
  IOSConfig.XcodeUtils.ensureGroupRecursively(project, 'Resources');

  const resourcesGroup = project.pbxGroupByName('Resources');
  const basename = path.basename(folderPath);
  const existingChild = resourcesGroup?.children?.find((child) => child.comment === basename);
  if (existingChild) return;

  const target = project.getTarget('com.apple.product-type.application');
  const file = new PbxFile(folderPath, {
    lastKnownFileType: 'folder',
    defaultEncoding: undefined,
  });
  file.target = target?.uuid;
  file.uuid = project.generateUuid();
  file.fileRef = project.generateUuid();
  file.fileEncoding = undefined;

  project.addToPbxFileReferenceSection(file);
  project.addToPbxBuildFileSection(file);
  project.addToPbxResourcesBuildPhase(file);
  resourcesGroup.children.push({
    value: file.fileRef,
    comment: file.basename,
  });
}
