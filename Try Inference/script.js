let model, scaler, labelMapping;

async function loadArtifacts() {
  model = await tf.loadGraphModel('model.json');
  console.log("Model loaded");

  scaler = await fetch('scaler.json').then(r => r.json());
  labelMapping = await fetch('labels.json').then(r => r.json());
  console.log("Scaler & Labels loaded");
}



function normalize(val, mean, std) {
  return (val - mean) / std;
}

// === Fitur waktu ===
function timeFeatures(dt) {
  const d = new Date(dt);
  const hour = d.getHours();
  const dow = d.getDay();
  const month = d.getMonth() + 1;
  return [
    hour, dow, month,
    Math.sin(2 * Math.PI * hour / 24),
    Math.cos(2 * Math.PI * hour / 24),
    Math.sin(2 * Math.PI * dow / 7),
    Math.cos(2 * Math.PI * dow / 7),
    Math.sin(2 * Math.PI * month / 12),
    Math.cos(2 * Math.PI * month / 12)
  ];
}

// === Fitur tambahan (diff) ===
// di sisi web tidak ada histori → isi 0
function extraFeatures() {
  return [0, 0, 0, 0];
}

// === Preprocessing time series ===
function preprocessTS(values, dt) {
  const timeFeats = timeFeatures(dt);
  const extraFeats = extraFeatures();
  let feats = [...values, ...timeFeats, ...extraFeats]; // total 18 fitur

  feats = feats.map((v, i) => normalize(v, scaler.mean[i], scaler.std[i]));
  const seq = Array(24).fill(feats);
  return tf.tensor([seq]); // [1,24,18]
}

// === Preprocessing gambar ===
async function preprocessImage(file) {
  const img = new Image();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      img.src = reader.result;
      img.onload = () => {
        let tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224,224])
          .toFloat()
          .div(255.0)
          .expandDims(0); // [1,224,224,3]
        resolve(tensor);
      };
    };
    reader.readAsDataURL(file);
  });
}

// === Inference ===
async function runInference() {
  const values = [
    parseFloat(document.getElementById("suhu").value),
    parseFloat(document.getElementById("kelembapan").value),
    parseFloat(document.getElementById("curah_hujan").value),
    parseFloat(document.getElementById("kecepatan_angin").value),
    parseFloat(document.getElementById("tutupan_awan").value)
  ];
  const dt = document.getElementById("datetime").value;
  const file = document.getElementById("imgInput").files[0];

  const tsTensor = preprocessTS(values, dt);
  const imgTensor = await preprocessImage(file);

  const prediction = await model.executeAsync({
    'img_input': imgTensor,
    'ts_input1': tsTensor,
    'ts_input2': tsTensor
  });

  const probs = await prediction.data();
  const predictedIndex = prediction.argMax(-1).dataSync()[0];
  const predictedLabel = labelMapping[predictedIndex];

  // Tambahkan keterangan waktu prediksi
  const inputTime = new Date(dt);
  const predictedTime = new Date(inputTime.getTime() + 60*60*1000); // +1 jam
const predictedTimeStr = predictedTime.toLocaleString("id-ID", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false
});


  let outputText = `Prediksi cuaca pada ${predictedTimeStr} (1 jam ke depan):\n\n`;
  outputText += "Probabilitas:\n";
  Object.entries(labelMapping).forEach(([i, lbl]) => {
    outputText += `${lbl}: ${(probs[i]*100).toFixed(2)}%\n`;
  });
  outputText += `\nPrediksi utama: ${predictedLabel}`;
  document.getElementById("output").textContent = outputText;
}

loadArtifacts().then(() => {
  document.getElementById("predictBtn").addEventListener("click", runInference);
});
