import React from "react";

/**
 * Interface that controls which properties are in the ProbabilitiesTracker props.
 * @param probabilities Object that is returned from the model API and parsed from JSON.
 */
interface IProps {
    probabilities: object
}

/**
 * Component that handles all functionality and UI regarding the probabilities that will be generated.
 */
class ProbabilitiesTracker extends React.Component<IProps> {
    /**
     * Constructor for the ProbabilitiesTracker component.
     * @param props Object that contains the properties for the ProbabilitiesTracker component.
     */
    constructor(props: IProps) {
        super(props);
    }

    /**
     * Function that renders this component.
     * @returns The JSX rendering of this component.
     */
    render() {
        var probs = this.props.probabilities
        return (
            <div className="probabilities-container">
                {Object.keys(probs).map((prob, i) => (
                    <div className="probability" style={{ backgroundColor: `rgba(255, 71, 71, ${probs[prob as keyof typeof probs] * 0.6})`}}>
                        <span className="probability-key">{i+1}. {prob} </span>
                        <span className="probability-value">{parseFloat(probs[prob as keyof typeof probs]).toFixed(5)}</span>
                    </div>
                ))}
            </div>
        )
    }

}

export default ProbabilitiesTracker;