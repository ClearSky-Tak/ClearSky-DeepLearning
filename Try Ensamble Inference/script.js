// script.js
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

function diffFeatures() {
  return [0, 0, 0, 0];
}

function windFeatures(kecepatan) {
  const wind_deg = 0;
  const wind_sin = 0;
  const wind_cos = 1;
  const wind_x = kecepatan * wind_cos;
  const wind_y = kecepatan * wind_sin;
  return [wind_deg, wind_sin, wind_cos, wind_x, wind_y];
}

function dayMask(dt) {
  const d = new Date(dt);
  const hour = d.getHours();
  const isDay = (hour >= 6 && hour < 18) ? 1.0 : 0.0;
  return tf.tensor([[isDay]]);
}

function preprocessTS(values, dt) {
  const [suhu, kelembapan, curah_hujan, kecepatan_angin, tutupan_awan] = values;

  const timeFeats = timeFeatures(dt);
  const diffFeats = diffFeatures();
  const windFeats = windFeatures(kecepatan_angin);

  let feats = [
    suhu, kelembapan, curah_hujan, kecepatan_angin, tutupan_awan,
    ...timeFeats,
    ...diffFeats,
    ...windFeats
  ];

  feats = feats.map((v, i) => normalize(v, scaler.mean[i], scaler.std[i]));
  const seq = Array(24).fill(feats);
  return tf.tensor([seq]);
}

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
          .expandDims(0);
        resolve(tensor);
      };
    };
    reader.readAsDataURL(file);
  });
}

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
  const maskTensor = dayMask(dt);

  const prediction = await model.executeAsync({
    'img_input': imgTensor,
    'ts_input': tsTensor,
    'mask_day': maskTensor
  });

  const probs = await prediction.data();
  const predictedIndex = prediction.argMax(-1).dataSync()[0];
  const predictedLabel = labelMapping[predictedIndex];

  const inputTime = new Date(dt);
  const predictedTime = new Date(inputTime.getTime() + 60*60*1000)
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
