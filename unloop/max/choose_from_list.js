subdivs = [0.125, 0.25, 0.5, 1, 2, 4];
subdivs = subdivs.map(function(x) { return x; });

function bang() {
    var i = Math.floor(Math.random() * subdivs.length);
    outlet(0, subdivs[i]);
}