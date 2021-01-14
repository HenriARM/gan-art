/**
 * @param {String} str - HTML representing a single element
 * @return {Node | null}
 */
function htmlToElement(str) {
    let template = document.createElement('template');
    str = str.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = str;
    return template.content.firstChild;
}

/**
 * @param {String} str - HTML representing any number of sibling elements
 * @return {NodeList}
 */
function htmlToElements(str) {
    let template = document.createElement('template');
    template.innerHTML = str;
    return template.content.childNodes;
}


//---------------------------- TODO: maybe separate help file

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

function createImageListItem(imageName, imageLink) {
    return htmlToElements('<li>' +
        '<input type="radio" name="select" id="' + imageName + '">' +
        '<div class="item-hugger">' +
        '<div class="title">' + imageName + '</div>' +
        '<img class="thumb-image" src="' + imageLink + '"/>' +
        '<label for="' + imageName + '"></label>' +
        '</div>' +
        '<div class="content">' +
        '<div class="item-wrapper">' +
        '<img src="' + imageLink + '"/>' +
        '<div class="title">' + imageName + '</div>' +
        '</div>' +
        '</div>' +
        '</li>');
}

function clearGallery() {
    let galleryList = document.getElementById("gallery");
    // delete all child elements
    galleryList.innerHTML = '';
}

function generateGallery() {

    // Get nodes which store form data
    const datasetType = document.getElementById("form-dataset-type").value;
    const gallerySize = document.getElementById("form-gallery-size").value;

    console.log(datasetType);
    console.log(gallerySize);

    if (datasetType === undefined || datasetType === null ||
        gallerySize === undefined || gallerySize === null) {
        console.log("error");
        // TODO
    }

    // check gallerySize is positive number
    if (gallerySize <= 0) {
        console.log("bad gallery size");
    }


    // build request
    const request = "http://0.0.0.0:4444/images?dataset=" + datasetType + "&size=" + gallerySize;
    console.log(request);

    let client = new HttpClient();
    client.get(request, function (response) {
        console.log(response);
        console.log(typeof response);

        let imagePayload = JSON.parse(response);
        console.log(imagePayload);
        console.log(typeof imagePayload);

        let galleryList = document.getElementById("gallery");

        for (let i = 0; i < imagePayload.length; i++) {
            let tuple = imagePayload[i];
            let imageName = tuple[0];
            let imageLink = tuple[1];

            console.log(imageName);
            console.log(imageLink);

            let imageItem = createImageListItem(imageName, imageLink);

            for (let j = 0; j < imageItem.length; j++) {
                galleryList.appendChild(imageItem[j]);

            }
            console.log(imageItem);
            console.log(typeof imageItem);

            // append image item to gallery

        }
    });

}


function generate() {
    clearGallery();
    generateGallery();
}
