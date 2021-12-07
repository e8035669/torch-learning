
async function file_selected(e) {
    let img = document.getElementById('selected-image');
    let file = document.getElementById('file-input').files[0];
    let url = URL.createObjectURL(file);
    let img_div = document.getElementById('img-display');
    console.log(file)
    console.log(url);
    img.src = url;
    img_div.style = "";

    let image_base64 = await toBase64(file);
    let base64_part_index = image_base64.indexOf(",") + 1;
    let base64_part = image_base64.slice(base64_part_index);
    // console.log(base64_part);

    let infer_url = "http://localhost:5000/invocations";
    var data = [
        { "image": base64_part }
    ];

    fetch(infer_url, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: new Headers({
            'Content-Type': 'application/json; format=pandas-records'
        })
    }).then(res => res.json())
        .catch(error => console.error('Error:', error))
        .then(response => console.log('Success', response));

}

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

