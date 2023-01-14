const stDoc = window.parent.document;

const container = stDoc.querySelector(".main > .block-container");

const buttons = Array.from(stDoc.querySelectorAll(".stButton > button"));
const download_button = buttons.find(el => el.innerText === "Download PDF");
const height = container.scrollHeight;
const width = container.scrollWidth;

let pdf_width = height + 20;
let pdf_height = pdf_width * 1.5 + 20;

let n_pages = Math.ceil(height / pdf_height) - 1;
download_button.innerText = "Downloading PDF...";

html2canvas(container, {allowTaint: true,foreignObjectRendering: true}).then(function (canvas) {{

    canvas.getContext('2d');

    let imgData = canvas.toDataURL("temp/jpeg", 1.0);
    let pdf = new jsPDF('p', 'px', [pdf_width, pdf_height]);
    pdf.addImage(imgData, 'JPG', 10, 10, width, height);

    for (let page = 1; page <= n_pages; page++) {{
        pdf.addPage();
        pdf.addImage(imgData, 'JPG', 10, -(pdf_height * page) + 40, width, height);
    }}

    pdf.save('test.pdf');

    download_button.innerText = "Download PDF";
}})
