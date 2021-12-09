
async function file_selected() {
    let files = document.getElementById('file-input').files;
    console.log('files', files)

    try {
        let input_data = await prepare_data(files);
        console.log(input_data);

        let result = await server_predict(input_data);
        console.log(result);

        let pred_result = [];
        for (let i = 0; i < files.length; ++i) {
            pred_result.push({
                'file': files[i],
                'pred': result[i]
            });
        }
        console.log(pred_result);

        let classes = new Set(pred_result.map(r => r['pred']['cls']));
        console.log(classes);

        let splited_result = {};
        for (let cls of classes) {
            let filtered = pred_result.filter(r => r['pred']['cls'] == cls);
            splited_result[cls] = filtered;
        }
        console.log(splited_result);

        generate_webview(splited_result);

    } catch (e) {
        console.log('Error', e);
    }
}

async function prepare_data(files) {
    let data = [];
    for (let f of files) {
        data.push({
            'image': await image_to_base64(f)
        })
    }
    return data;
}


function image_to_base64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            let result = reader.result;
            let data_index = result.indexOf(',') + 1;
            let data_part = result.slice(data_index);
            resolve(data_part);
        }
        reader.onerror = error => reject(error);
    });
}

async function server_predict(data) {
    let infer_url = "/invocations";
    return fetch(infer_url, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: new Headers({
            'Content-Type': 'application/json; format=pandas-records'
        })
    }).then(res => res.json());
}

function generate_webview(pred_result) {
    let display = document.getElementById('img-display');
    let gen_html = '';

    for (let cls in pred_result) {
        let cls_name = pred_result[cls][0]['pred']['cls_name'];
        gen_html += '<hr /><h2>Class: ' + cls_name + '</h2>';
        gen_html += '<div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">';

        let preds = pred_result[cls];
        for (let pred of preds) {
            let f = pred['file'];
            let url = URL.createObjectURL(f);
            let fname = f.name;
            // gen_html += `
            // <div class="col">
            //     <img style="height: 200px" src="${url}" class="img-thumbnail"></img>
            //     <p>${f.name}</p>
            // </div>
            // `;
            gen_html += `
            <div class='col'>
                <div class='card shadow-sm'>
                    <img class='card-img-top' width='100%' height='225' style='object-fit: contain' src="${url}"></img>
                    <div class='card-body'>
                        <p class='card-text'>${f.name}</p>
                    </div>
                </div>
            </div>
            `;
        }
        gen_html += '</div>';

    }

    display.innerHTML = gen_html;
}


function test_button() {
    let display = document.getElementById('img-display');
    let gen_html = '<h2>Good</h2><hr></hr><h2>Bad</h2>'

    display.innerHTML = gen_html;


    console.log('Here');

}
