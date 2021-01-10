let HttpClient = function () {
    this.get = function (aUrl, aCallback) {
        let asyncHttpRequest = new XMLHttpRequest();
        asyncHttpRequest.onreadystatechange = function () {
            if (asyncHttpRequest.readyState === 4 && asyncHttpRequest.status === 200)
                aCallback(asyncHttpRequest.responseText);
        };

        asyncHttpRequest.open("GET", aUrl, true);
        asyncHttpRequest.send(null);
    }
};

function connectMinio() {

}

function doSmth() {
    console.log('Hello World');

    let client = new HttpClient();
    client.get('http://0.0.0.0:4444/images/?size=64', function (response) {
        let image = new Image();
        image.src = 'data:image/png;base64,' + response;
        document.body.appendChild(image);
    });
}



