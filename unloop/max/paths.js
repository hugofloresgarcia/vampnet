var pathModes = ["off", "wander", "circle", "bounce"];

// Define state object
var state = {
    coords: [],
    coordidx: 0,
    mode: "off"
};

// init with a random 
setPath("random");

// the space ranges from -1 to 1 in x and y, z must always be 0
function bang() {
    // TODO: emit next xyz coordinate in path
    if (state.coords.length > 0) {
        outlet(0, state.coords[state.coordidx]);
        state.coordidx = (state.coordidx + 1) % state.coords.length;
    }
    else {
        post("no path to follow\n");
    }
}

function setPath(mode) {
    if (pathModes.indexOf(mode) >= 0) {
        state.mode = mode;
    }
    state.mode = mode;

    // generate points for the pathe
    if (state.mode == "circle") {
        // circle around in a random direction
        state.coords = [];
        var numPoints = Math.round(Math.random() * 100);
        var angle = Math.random() * 2 * Math.PI;

        var direction = Math.random() < 0.5 ? 1 : -1;
        for (var i = 0; i < numPoints; i++) {
            state.coords.push([Math.cos(angle), Math.sin(angle), 0]);
            angle += 2 * Math.PI / numPoints * direction;
        }
    }
    else if (state.mode == "wander") {
            // wander around in brownian motion
            state.coords = [];
            var numPoints = Math.round(Math.random() * 100);
            // var x = 0;
            // var y = 0;
            // pick a random starting point within -1 and 1
            var x = Math.random() * 2 - 1;
            var y = Math.random() * 2 - 1;
            for (var i = 0; i < numPoints; i++) {
                x += Math.random() * 0.2 - 0.1; // TODO: this 0.1 controls wander amt
                y += Math.random() * 0.2 - 0.1;

                // clamp to -1 to 1
                x = Math.min(1, Math.max(-1, x));
                y = Math.min(1, Math.max(-1, y));
                state.coords.push([x, y, 0]);
            }
        }
    else if (state.mode == "bounce") {
        // bounce around two points
        state.coords = [];
        var numPoints = 2;
        var x = 0;
        var y = 0;
        
        // pick two random quadrants to place the point in
        quads = {
            1: [1, 1],
            2: [-1, 1],
            3: [-1, -1],
            4: [1, -1]
        }
        var quadindices = [1, 2, 3, 4];
        // scramble quadindices 
        quadindices.sort(function(a, b) { return Math.random() - 0.5; });
        var quadidx1 = quadindices.pop();
        var quadidx2 = quadindices.pop();
        var quad1 = quads[quadidx1];
        var quad2 = quads[quadidx2];
        // post("quad1: " + quad1 + " quad2: " + quad2);

        // pick point 1, a random point in the range (0, 1), then scale by the quad
        var x1 = Math.random() * quad1[0];
        var y1 = Math.random() * quad1[1];

        // pick point 2, a random point in the range (0, 1), then scale by the quad
        var x2 = Math.random() * quad2[0];
        var y2 = Math.random() * quad2[1];

        // generate the path
        state.coords.push([x1, y1, 0]);
        state.coords.push([x2, y2, 0]);
    }
    else if (state.mode == "random") {
        state.coords = [];
        var numPoints = Math.round(Math.random() * 100) + 4;
        for (var i = 0; i < numPoints; i++) {
            state.coords.push([Math.random() * 2 - 1, Math.random() * 2 - 1, 0]);
        }
        // post("random now has " + state.coords.length + " points\n");
    }
    else {
        post("unknown path mode");
    }
}
