import Sound from 'react-native-sound';
import {createRNSoundPlayer} from '@kittentts/react-native';

export function createBenchmarkPlayer() {
  return createRNSoundPlayer(Sound);
}
