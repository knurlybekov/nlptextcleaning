import React, {useState} from "react"; 

function ClassificationInput(){
    const [input, changeInput] = useState("");
    const [guess, changeGuess] = useState("");

    function sendText(){
        let sendUrl = `http://cs.tru.ca:8010`;
        let text = {"inputText": input};
        fetch(sendUrl,{
            method:'POST',
            headers:{'Content-Type': 'application/json'},
            body: JSON.stringify(text)
        })
        .then(response => response.json())
        .then(data => changeGuess(data))
        .catch((error) => console.log(error));
    }

    return(
        <>
        <div class = "form-group">
        <label for = "inputText">Text to be guessed: </label>
        <input name = "inputText" class = "form-control" type = "text" onInput = {(data) => {changeInput(data.target.value);}} value = {input}></input>
        </div>

        <div class = "form-group">
        <button class = "btn btn-primary" onClick = {sendText}>Send</button>
        </div>

        <div class = "form-group">
        <label for = "result">Result: </label>
        <input name = "result" class = "form-control" type = "text" readOnly value = {guess}></input>
        </div>
        </>
    )
}

export {ClassificationInput};