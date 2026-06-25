/**
 * @format
 */

import 'react-native';
import React from 'react';

// Note: import explicitly to use the types shipped with jest.
import {it, jest} from '@jest/globals';

jest.mock('react-native-sound', () => jest.fn());

jest.mock('@kittentts/react-native', () => {
  const KittenModel = {
    Nano: 'nano',
    NanoInt8: 'nano-int8',
    Micro: 'micro',
    Mini: 'mini',
  };
  const KittenVoice = {
    Bella: 'bella',
    Nova: 'nova',
  };

  return {
    KittenTTS: {
      create: jest.fn().mockResolvedValue({
        dispose: jest.fn().mockResolvedValue(undefined),
        generate: jest.fn(),
        speak: jest.fn(),
      }),
    },
    KittenModel,
    KittenVoice,
    ALL_VOICES: [KittenVoice.Bella, KittenVoice.Nova],
    createRNSoundPlayer: jest.fn(() => ({})),
    modelDisplayName: (model: string) => model,
    voiceDisplayName: (voice: string) => voice,
  };
});

import App from '../App';

// Note: test renderer must be required after react-native.
import renderer, {act} from 'react-test-renderer';

it('renders correctly', async () => {
  await act(async () => {
    renderer.create(<App />);
  });
});
