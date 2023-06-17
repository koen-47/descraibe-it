import React from 'react';
import ReactDOM from 'react-dom/client';
import './static/css/index.scss';
import App from './App';

/**
 * Function that contains the root element of the app.
 */
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

/**
 * Render all the components that can be found in the root.
 */
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
