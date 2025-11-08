const form = document.getElementById("uploadForm");
const result = document.getElementById("result");
const preview = document.getElementById("preview");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("fileInput").files[0];
  if (!file) {
    alert("Please select an image!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  result.innerHTML = "Analyzing...";
  preview.style.display = "none";

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (data.error) {
    result.innerHTML = `<span style="color:red;">${data.error}</span>`;
    return;
  }

  preview.src = data.image_url;
  preview.style.display = "block";
  result.innerHTML = `Prediction: <b>${data.label}</b> <br> Confidence: ${data.confidence}%`;
});
