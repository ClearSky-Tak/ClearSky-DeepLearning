// index.js (versi Node.js)
const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// Global vars
let model, scaler, labelMapping;

// === Load artifacts ===
async function loadArtifacts() {
  model = await tf.loadGraphModel('file://model/model.json');
  console.log("Model loaded");

  scaler = JSON.parse(fs.readFileSync('scaler.json'));
  labelMapping = JSON.parse(fs.readFileSync('labels.json'));
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

// === Preprocessing gambar (Node.js pakai tfjs-node) ===
async function preprocessImage(path) {
  const sharp = require('sharp'); // pastikan install sharp
  const imageBuffer = await sharp(path)
    .resize(224, 224)
    .toFormat('png')
    .toBuffer();

  const { Image } = require('image-js');
  const img = await Image.load(imageBuffer);

  let tensor = tf.tensor3d(img.data, [img.height, img.width, 4])
    .slice([0,0,0],[img.height,img.width,3]) // ambil RGB
    .toFloat()
    .div(255.0)
    .expandDims(0); // [1,224,224,3]

  return tensor;
}

// === Inference ===
async function runInference() {
  // contoh input manual
  const values = [26.5, 80, 0, 3.2, 50]; // suhu, kelembapan, dll
  const dt = new Date(); // waktu sekarang
  const imgPath = 'sample.png'; // path gambar awan

  const tsTensor = preprocessTS(values, dt);
  const imgTensor = await preprocessImage(imgPath);
  const maskTensor = dayMask(dt);

  const prediction = await model.executeAsync({
    'img_input': imgTensor,
    'ts_input': tsTensor,
    'mask_day': maskTensor
  });

  const probs = await prediction.data();
  const predictedIndex = prediction.argMax(-1).dataSync()[0];
  const predictedLabel = labelMapping[predictedIndex];

  const predictedTime = new Date(dt.getTime() + 60*60*1000);
  const predictedTimeStr = predictedTime.toLocaleString("id-ID", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
  });

  console.log(`Prediksi cuaca pada ${predictedTimeStr} (1 jam ke depan):`);
  Object.entries(labelMapping).forEach(([i, lbl]) => {
    console.log(`${lbl}: ${(probs[i]*100).toFixed(2)}%`);
  });
  console.log(`Prediksi utama: ${predictedLabel}`);
}

// === Main ===
loadArtifacts().then(runInference);



// npm install @tensorflow/tfjs @tensorflow/tfjs-node sharp image-js
// node index.js
