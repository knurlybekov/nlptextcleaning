import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';
import {ClassificationInput} from "./module.js";

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <>
    
  <div id = 'appBox'>
    <h1 id = "header">Text Classifier</h1>
  </div>
  <div class = 'panel panel-default'>
    <div class = "panel-heading"><h3>Echo</h3></div>
    <div class = "panel-body">
    <ClassificationInput/>
    </div>
  </div>
    </>
);
reportWebVitals();
