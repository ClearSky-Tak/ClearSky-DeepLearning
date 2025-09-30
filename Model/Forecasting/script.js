let model;
let scaler;
let labelMapping;

async function loadArtifacts() {
  // Abaikan regularizer L2
  model = await tf.loadLayersModel('model.json', { strict: false });
  console.log("Model loaded");

  scaler = await fetch('scaler.json').then(r => r.json());
  labelMapping = await fetch('labels.json').then(r => r.json());

  console.log("Scaler loaded:", scaler);
  console.log("Labels loaded:", labelMapping);
}

function timeFeatures(dt) {
  const d = new Date(dt);
  const hour = d.getHours();
  const dow = d.getDay();
  const month = d.getMonth() + 1;

  return [
    Math.sin(2 * Math.PI * hour / 24),
    Math.cos(2 * Math.PI * hour / 24),
    Math.sin(2 * Math.PI * dow / 7),
    Math.cos(2 * Math.PI * dow / 7),
    Math.sin(2 * Math.PI * month / 12),
    Math.cos(2 * Math.PI * month / 12)
  ];
}

// Tambahan 5 fitur dari add_berawan_features (isi 0 untuk real-time)
function extraFeatures(values) {
  const rh_diff3 = 0;
  const rh_diff6 = 0;
  const temp_diff3 = 0;
  const temp_diff6 = 0;
  const dummy_extra = 0; // fitur ke-18 (misalnya press_diff3 atau rad_std jika ada di training)

  return [rh_diff3, rh_diff6, temp_diff3, temp_diff6, dummy_extra];
}

function normalize(val, mean, std) {
  return (val - mean) / std;
}

function preprocessInput(values, dt) {
  const timeFeats = timeFeatures(dt);   // 6 fitur
  const extraFeats = extraFeatures(values); // 5 fitur
  let feats = [...values, ...timeFeats, ...extraFeats]; // total 18

  if (!scaler || !scaler.mean || !scaler.std) {
    throw new Error("Scaler belum siap");
  }

  feats = feats.map((v, i) => normalize(v, scaler.mean[i], scaler.std[i]));

  const seq = Array(24).fill(feats);
  return tf.tensor([seq]); // [1,24,18]
}

async function runInference(values, dt) {
  const inputTensor = preprocessInput(values, dt);

  // Model punya 2 input branch → duplikat tensor
  const prediction = model.predict([inputTensor, inputTensor]);

  const probs = await prediction.data();
  const predictedClass = prediction.argMax(-1).dataSync()[0];
  return { probs: Array.from(probs), predictedClass };
}

async function init() {
  await loadArtifacts();
  console.log("Artifacts siap, form bisa dipakai");

  document.getElementById("predictBtn").addEventListener("click", async () => {
    const values = [
      parseFloat(document.getElementById("suhu").value),
      parseFloat(document.getElementById("kelembapan").value),
      parseFloat(document.getElementById("curah_hujan").value),
      parseFloat(document.getElementById("kecepatan_angin").value),
      parseFloat(document.getElementById("arah_angin").value),
      parseFloat(document.getElementById("tutupan_awan").value),
      parseFloat(document.getElementById("jarak_pandang").value)
    ];
    const dt = document.getElementById("datetime").value;

    try {
      const result = await runInference(values, dt);
      document.getElementById("output").textContent =
        "Probabilitas kelas: " + JSON.stringify(result.probs, null, 2) +
        "\nPrediksi kelas: " + labelMapping[result.predictedClass];
    } catch (err) {
      console.error(err);
      document.getElementById("output").textContent = "Error: " + err.message;
    }
  });
}

init();
