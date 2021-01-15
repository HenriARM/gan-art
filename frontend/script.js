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

function copyImage(imageName, imageLink) {
    console.log('Click');
    // console.log(this);
    console.log(imageName);
    console.log(imageLink);

    let whitebox = document.getElementById("whitebox");
    whitebox.innerHTML = "";

    let childNodes = htmlToElements(
        '<div class="content">' +
        '<div class="item-wrapper d-flex flex-column justify-content-center">' +
        '<div class="title" style="align-self: center">' + imageName + '</div>' +
        '<img src="' + imageLink + '" width="80%" height="80%"/>' +
        '</div>' +
        '</div>'
    );

    for (let i = 0; i < childNodes.length; i++) {
        let child = childNodes[i];
        whitebox.appendChild(child);
    }

}

const lastTwoImages = [];
let interImages;
let datasetType;

function saveLastTwoImages(elem, imageName, imageLink) {

    console.log('Double click');

    if (lastTwoImages.length < 2) {
        lastTwoImages.push([elem, imageName, imageLink]);
        elem.style.background = '#ffbf9b';
    } else {
        // get oldest element
        let oldE = lastTwoImages.shift()[0];
        oldE.style.background = "none";


        lastTwoImages.push([elem, imageName, imageLink]);
        elem.style.background = '#ffbf9b';
    }

    console.log(lastTwoImages);
}

function interpolate() {
    console.log(datasetType);

    if (lastTwoImages.length !== 2) {
        console.log('Error, not enough selected images');
        return 0;
    }

    // build request
    // send 2 image names
    const request = "http://0.0.0.0:4444/interpolate?img1=" + lastTwoImages[0][1]
        + "&img2=" + lastTwoImages[1][1] + "&dataset=" + datasetType;
    console.log(request);

    let client = new HttpClient();
    client.get(request, function (response) {
        console.log(response);
        console.log(typeof response);

        let imagePayload = JSON.parse(response);
        console.log(imagePayload);
        console.log(typeof imagePayload);

        interImages = imagePayload;

        // for (let i = 0; i < imagePayload.length; i++) {
        //     let tuple = imagePayload[i];
        //     let imageName = tuple[0];
        //     let imageLink = tuple[1];
        //
        //     console.log(imageName);
        //     console.log(imageLink);
        // }

        // TODO: image source put first

        let interImage = document.getElementById("inter-image");
        interImage.src = imagePayload[0][1];
        // TODO: show slider
    });
}

function createImageListItem(imageName, imageLink) {
    return htmlToElements('<li onclick=\"copyImage(\'' + imageName + '\',\'' + encodeURI(imageLink) + '\'' + ')"' +
        ' ondblclick=\"saveLastTwoImages(this, \'' + imageName + '\',\'' + encodeURI(imageLink) + '\'' + ')"' +
        '<input type="radio" name="select" id="' + imageName + '">' +
        '<div class="item-hugger">' +
        '<div class="title">' + imageName + '</div>' +
        '<img class="thumb-image" src="' + imageLink + '"/>' +
        '<label for="' + imageName + '"></label>' +
        '</div>' +
        '</div>' +
        '</li>');


    // TODO: All 4 variants didnt worked
    // '<iframe id="invisible" style="display:none;"></iframe>\n' +

    // '<form method="get" action="' + imageLink + '">' +
    // '<button type="submit">Download!</button>' +
    // '</form>' +


    // '<div class="title">' +
    // '<a href="' + imageLink + '" download="' + imageName + '">' +
    // imageName +
    // '</div>' +


    // '<a href="' + imageLink + '" download="' + imageName + '">' +
    // '<button type="submit" onclick="window.open("' + imageLink + '")">Download!</button>' +
    // '</a>' +

}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function clearGallery() {
    let galleryList = document.getElementById("gallery");
    // delete all child elements
    galleryList.innerHTML = '';
}

function updateSlider(interStep) {
    console.log(interStep);
    let interImage = document.getElementById("inter-image");
    interImage.src = interImages[interStep][1];
}

function createLoadingIcon() {
    let loadingIcon = htmlToElement('<div id="loading-icon" class="lds-dual-ring"></div>');
    let whitebox = document.getElementById("whitebox");
    whitebox.parentNode.insertBefore(loadingIcon, whitebox);
}

function removeLoadingIcon() {
    let loadingIcon = document.getElementById("loading-icon");
    loadingIcon.outerHTML = "";
}

async function generateGallery() {
    createLoadingIcon();
    await sleep(2000);

    // Get nodes which store form data
    datasetType = document.getElementById("form-dataset-type").value;
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

        if (imagePayload.length) removeLoadingIcon();

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
        }
    });

}

function clearWhitebox() {
    let whitebox = document.getElementById("whitebox");
    whitebox.innerHTML = '';
}


function generate() {
    clearGallery();
    clearWhitebox();
    generateGallery();

}

// TODO: create and remove slider with image tag