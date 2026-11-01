require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

# Kitten TTS 2's language model can run on the GPU through MLX. MLX needs iOS 17
# and adds to the app's size, so it is built only when the app asks:
# `KITTENTTS_MLX=1 pod install`, or the Expo plugin's `mlx: true`, which writes
# "kittentts.mlx": "true" to ios/Podfile.properties.json. Without it this pod
# has no sources (only its React-Core dependency, so CocoaPods accepts it) and
# the SDK runs everything through ONNX Runtime.
mlx_enabled =
  if ENV.key?('KITTENTTS_MLX')
    ENV['KITTENTTS_MLX'] == '1'
  else
    begin
      properties = File.join(Pod::Config.instance.installation_root.to_s, 'Podfile.properties.json')
      File.exist?(properties) && JSON.parse(File.read(properties))['kittentts.mlx'].to_s == 'true'
    rescue StandardError
      false
    end
  end

Pod::Spec.new do |s|
  s.name = 'KittenTTSReactNative'
  s.version = package['version']
  s.summary = package['description']
  s.homepage = package['homepage']
  s.license = package['license']
  s.authors = package['author']
  s.source = { git: package['repository']['url'], tag: "v#{s.version}" }

  s.dependency 'React-Core'

  if mlx_enabled
    s.platforms = { ios: '17.0' }
    s.source_files = 'ios/**/*.{h,m,swift}'
    s.swift_version = '5.9'
    s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
    spm_dependency(s,
      url: 'https://github.com/ml-explore/mlx-swift',
      requirement: { kind: 'exactVersion', version: '0.33.0' },
      products: ['MLX'])
  else
    s.platforms = { ios: '13.4' }
  end
end
