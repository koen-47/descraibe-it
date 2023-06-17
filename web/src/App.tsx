import React from 'react';

import './App.css';
import Main from "./layouts/Main"
import GameContainer from './components/GameContainer'

/**
 * Component that renders the entire web app.
 */
function App() {
  return (
    <div className="App">
      <Main>
        <GameContainer />
      </Main>
    </div>
  );
}


export default App;
