import React from 'react';

/**
 * Interface that controls which properties are in the Main props.
 * @param children All JSX elements that will be rendered in this component.
 */
type Props = {
    children?: React.ReactNode
}

/**
 * Component that contains the current view of the app.
 */
const Main: React.FunctionComponent<Props> = ({ children } : Props) => {
    return (
        <div id="wrapper">
            <div id="main-container">
                <div id="main">
                    <header id="main-header">
                        <p>descr<span>AI</span>be it</p>
                    </header>
                    
                    <div id="main-content">
                        <div id="main-content-body">{children}</div>
                    </div>

                    <footer id="main-footer">

                    </footer>
                </div>
            </div>
            
        </div>
    )
};

export default Main;