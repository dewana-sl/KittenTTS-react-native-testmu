import React from 'react';
import {AppRegistry} from 'react-native';
import App from './App';
import {name as appName} from './app.json';

type WebBenchmarkState = {
  bundleLoaded?: boolean;
  appMounted?: boolean;
  error?: string | null;
  updatedAt?: string;
};

function publishWebBenchmarkState(partial: WebBenchmarkState) {
  const scope = globalThis as {
    __KITTEN_BENCHMARK_STATE__?: WebBenchmarkState;
    __KITTEN_BENCHMARK_ERROR__?: string;
  };
  scope.__KITTEN_BENCHMARK_STATE__ = {
    ...(scope.__KITTEN_BENCHMARK_STATE__ ?? {}),
    ...partial,
    updatedAt: new Date().toISOString(),
  };
  if (partial.error) {
    scope.__KITTEN_BENCHMARK_ERROR__ = partial.error;
  }
}

publishWebBenchmarkState({bundleLoaded: true});

window.addEventListener('error', event => {
  publishWebBenchmarkState({
    error: event.error instanceof Error ? event.error.message : event.message,
  });
});

window.addEventListener('unhandledrejection', event => {
  const reason = event.reason;
  publishWebBenchmarkState({
    error: reason instanceof Error ? reason.message : String(reason),
  });
});

AppRegistry.registerComponent(appName, () => App);
AppRegistry.runApplication(appName, {
  initialProps: {},
  rootTag: document.getElementById('root'),
});
