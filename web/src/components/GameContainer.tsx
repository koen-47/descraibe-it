import React from 'react';

import { ImSpinner2 } from "react-icons/im"
import { AiOutlineEnter } from "react-icons/ai"

import ProbabilitiesTracker from './ProbabilitiesTracker';
import categories from "../data/categories_25.json";

interface IProps {
}

/**
 * Interface that controls which properties are in the GameContainer state.
 * @param description String that contains the current text from the input bar.
 * @param isRequesting Boolean that checks if the app is currently awaiting a response from a previously sent request to the API.
 * @param currentCategory String that contains the current category (word) that the user must describe.
 * @param possibleCategories Array of strings that contains the pool of possible categories (words) that may be randomly selected.
 * @param score Integer that contains the current score for the player.
 * @param currentPrediction Object that contains the response from the model API.
 * @param isShowingPopup Boolean that determines if a popup is being shown the the user.
 * @param popupText String that contains the information being shown the user during a popup.
 */ 
interface IState {
    description: string,
    isRequesting: boolean,
    currentCategory: string,
    possibleCategories: Array<string>,
    score: number,
    currentPrediction: object,
    isShowingPopup: boolean,
    popupText: string
}

/**
 * Component that renders all the necessary functionality and UI related to the game portion of the web app.
 */
class GameContainer extends React.Component<IProps, IState> {
    /**
     * Constructor for the GameContainer component.
     * @param props Object that contains the properties of the GameContainer component.
     */
    constructor(props: IProps) {
        super(props);
        this.state = {
            description: "",
            isRequesting: false,
            currentCategory: "",
            possibleCategories: categories,
            score: 0,
            currentPrediction: {},
            isShowingPopup: false,
            popupText: "Correct!"
        }
        this.handleSubmit = this.handleSubmit.bind(this);
        
    }

    /**
     * Function that is called when the component sucessfully mounts.
     */
    componentDidMount(): void {
        this.setState({
            currentCategory: this.state.possibleCategories[Math.floor(Math.random() * this.state.possibleCategories.length)]
        })
    }

    /**
     * Function that randomly selects a new category from the pool of available categories.
     */
    setRandomCategory() {
        var categories = this.state.possibleCategories
        var randomCategory = categories[Math.floor(Math.random() * categories.length)]
        this.setState({currentCategory: randomCategory})
    }

    /**
     * Function that handles the user's submission to the API to be passed through to the model.
     * @param e Event listener object.
     */
    handleSubmit(e: any) {
        e.preventDefault()
        if (this.state.description.includes(this.state.currentCategory)) {
            this.setState({isShowingPopup: true})
            this.setState({popupText: "Your answer cannot contain the given word!"})
            setTimeout(() => {
                this.setState({isShowingPopup: false})
            }, 2000);
        } else {
            this.setState({isShowingPopup: false})
            this.predict(this.state.description)
        }
    }
    
    /**
     * Function that handles sending the request to the model API with the raw text from the input bar as JSON.
     * @param description The raw text to be sent with the request.
     */
    predict(description: string) {
        this.setState({isRequesting: true})
        fetch("https://koen-kraaijveld.onrender.com/descraibeit", {
          method: "POST",
          body: JSON.stringify({"text": description}),
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          }
        })
        .then(response => 
            response.json().then(data => ({
              data: data,
              status: response.status
        })))
        .then(res => {
            const highestProbLabel = Object.keys(res.data)[0]

            if (highestProbLabel == this.state.currentCategory) {
                this.setState({score: this.state.score + 1})
                this.setRandomCategory()
                this.setState({isShowingPopup: true})
                this.setState({popupText: "Correct!"})
                setTimeout(() => {
                    this.setState({isShowingPopup: false})
                }, 2000);
            } else {
                this.setState({score: 0})
                this.setState({isShowingPopup: true})
                this.setState({popupText: "Game over!"})
                setTimeout(() => {
                    this.setState({isShowingPopup: false})
                }, 2000);
            }
            this.setState({isRequesting: false})
            this.setState({currentPrediction: res.data})
            console.log(res.data)
        })
    }
    
    /**
     * Function that renders this component.
     * @returns The JSX rendering of this component.
     */
    render() {
        return (
            <div>
                <div className="description-input-wrapper">
                    <div className="description-input-container">
                        {
                        !this.state.isShowingPopup && 
                        <div>
                            <div className="score-container">
                                <p className="game-info">Score:</p>
                                <p>{this.state.score}</p>
                            </div>
                            <div className="category-container">
                                <p className="game-info">Describe the word:</p>
                                <p>{this.state.currentCategory}</p>
                            </div>
                        </div>
                        }

                        {
                        this.state.isShowingPopup &&
                        <div className="popup-text-container"> 
                            {this.state.popupText}
                        </div>
                        }
                        
                        <div className="input-form-container">
                            <form onSubmit={this.handleSubmit}>
                                <input type="text" onChange={(e) => this.setState({description: e.target.value})} disabled={this.state.isRequesting}/>
                                <button id="btn-submit" onClick={this.handleSubmit} disabled={this.state.isRequesting}>
                                    {this.state.isRequesting ? 
                                    <ImSpinner2 className="spinning" size={30}/> :
                                    <AiOutlineEnter size={30}/>}
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
                <ProbabilitiesTracker probabilities={this.state.currentPrediction}/>
            </div>
            
        )
    }
    
}

export default GameContainer;