// assets/clientside.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: { // This 'clientside' will be your namespace
	getDimensions: function(id) {
	    const graphDiv = document.getElementById(id);
	    if (graphDiv) {
		const originalVisibility = graphDiv.style.visibility;
		graphDiv.style.visibility = 'visible';
		
		const dimensions = {
		    width: graphDiv.clientWidth,
		    height: graphDiv.clientHeight
		};
		
		graphDiv.style.visibility = originalVisibility;
		return dimensions;
	    }
	    return window.dash_clientside.no_update;
	}
    }
});
