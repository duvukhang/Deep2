const socket = io();

let requested = false;
let map = null;
let userMarker = null;
let poiMarkers = [];
let lastDrowsyTime = 0;

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}

function setStatusBox(state, isDrowsy) {
  const status = document.getElementById("status");

  if (!status) return;

  if (state === "DROWSY_CONFIRMED" || isDrowsy) {
    status.innerText = "NGỦ GẬT!";
    status.className = "alert-box alert alert-danger text-center";
  } else if (state === "WARNING_SUNGLASSES_MODE") {
    status.innerText = "CẢNH BÁO - KHÔNG THẤY MẮT";
    status.className = "alert-box alert alert-warning text-center";
  } else if (state === "SUNGLASSES_MODE") {
    status.innerText = "CHẾ ĐỘ KÍNH RÂM";
    status.className = "alert-box alert alert-info text-center";
  } else if (state === "WARNING_LEVEL_1") {
    status.innerText = "CÓ DẤU HIỆU MỆT";
    status.className = "alert-box alert alert-warning text-center";
  } else if (state === "CAMERA_BAD") {
    status.innerText = "GÓC CAMERA KÉM";
    status.className = "alert-box alert alert-primary text-center";
  } else if (state === "NO_FACE") {
    status.innerText = "KHÔNG THẤY KHUÔN MẶT";
    status.className = "alert-box alert alert-secondary text-center";
  } else {
    status.innerText = "TỈNH TÁO";
    status.className = "alert-box alert alert-success text-center";
  }
}

function initMap(lat, lon) {
  if (map !== null) {
    map.setView([lat, lon], 15);
    return;
  }

  map = L.map("map").setView([lat, lon], 15);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
}

function updateUserMarker(lat, lon, label = "Vị trí hiện tại") {
  if (!map) return;

  if (userMarker) {
    map.removeLayer(userMarker);
  }

  userMarker = L.marker([lat, lon])
    .addTo(map)
    .bindPopup(label)
    .openPopup();
}

function clearPoiMarkers() {
  if (!map) return;

  poiMarkers.forEach(marker => {
    map.removeLayer(marker);
  });

  poiMarkers = [];
}

function addPoiMarker(stop) {
  if (!map || !stop.lat || !stop.lon) return;

  const popupHtml = `
    <b>${stop.name}</b><br>
    ${stop.type}<br>
    Cách khoảng ${stop.distance_text}<br>
    <a href="${stop.direction_url}" target="_blank">Chỉ đường</a>
  `;

  const marker = L.marker([stop.lat, stop.lon])
    .addTo(map)
    .bindPopup(popupHtml);

  poiMarkers.push(marker);
}

async function getLocationByBrowser() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Trình duyệt không hỗ trợ geolocation"));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      position => {
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          source: "GPS trình duyệt"
        });
      },
      error => {
        reject(error);
      },
      {
        enableHighAccuracy: true,
        timeout: 6000,
        maximumAge: 30000
      }
    );
  });
}

async function getLocationByIP() {
  const res = await fetch("/api/location");
  const data = await res.json();

  if (!data.success) {
    throw new Error(data.message || "Không lấy được vị trí IP");
  }

  return {
    latitude: data.location.lat,
    longitude: data.location.lon,
    source: "IP máy tính",
    city: data.location.city,
    region: data.location.region,
    country: data.location.country
  };
}

async function getBestLocation() {
  try {
    return await getLocationByBrowser();
  } catch (err) {
    console.warn("Không lấy được GPS, fallback sang IP:", err);
    return await getLocationByIP();
  }
}

async function requestNearbyStops() {
  try {
    setText("location_text", "Đang lấy vị trí...");

    const radiusSelect = document.getElementById("radius_select");
    const radius = radiusSelect ? parseInt(radiusSelect.value) : 40000;

    const coords = await getBestLocation();

    const lat = coords.latitude;
    const lon = coords.longitude;

    initMap(lat, lon);

    const label = coords.city
      ? `Vị trí theo ${coords.source}: ${coords.city}, ${coords.country}`
      : `Vị trí theo ${coords.source}`;

    updateUserMarker(lat, lon, label);

    setText(
      "location_text",
      `${label} | lat: ${lat.toFixed(5)}, lon: ${lon.toFixed(5)}`
    );

    socket.emit("find_stops_request", {
      lat: lat,
      lon: lon,
      radius: radius
    });

  } catch (err) {
    console.error("Lỗi lấy vị trí:", err);
    setText("location_text", "Không lấy được vị trí. Hãy bật quyền vị trí hoặc kiểm tra mạng.");
  }
}

socket.on("update_data", data => {
  const ear = Number(data.ear || 0);
  const mar = Number(data.mar || 0);
  const pose = Number(data.head_pose || 0);
  const score = Number(data.drowsy_score || 0);
  const confidence = Number(data.confidence || 0);
  const threshold = Number(data.threshold || 0);
  const fallbackScore = Number(data.fallback_score || 0);
  const headDownDuration = Number(data.head_down_duration || 0);
  const fallbackYawnDuration = Number(data.fallback_yawn_duration || 0);

  setText("ear_val", ear.toFixed(2));
  setText("threshold_val", threshold.toFixed(2));
  setText("mar_val", mar.toFixed(2));
  setText("pose_val", pose.toFixed(2));
  setText("score_val", score.toFixed(2));
  setText("state_val", data.state || "NORMAL");
  setText("face_count", data.face_count ?? 0);
  setText("visible_eye_count", data.visible_eye_count ?? 0);
  setText("confidence_val", confidence.toFixed(2));
  setText("leaning_val", data.is_leaning ? "Có" : "Không");
  setText("occlusion", data.eye_visible ? "Không" : "Có");

  setText("detection_mode_val", data.detection_mode || "EYE_MODE");
  setText("fallback_score_val", fallbackScore.toFixed(2));
  setText("head_down_duration_val", headDownDuration.toFixed(2) + "s");
  setText("fallback_yawn_duration_val", fallbackYawnDuration.toFixed(2) + "s");

  setStatusBox(data.state, data.is_drowsy);

  if (data.is_drowsy) {
    const now = Date.now();

    if (!requested || now - lastDrowsyTime > 30000) {
      requestNearbyStops();
      requested = true;
      lastDrowsyTime = now;
    }
  }

  if (data.state === "NORMAL" || data.state === "SUNGLASSES_MODE") {
    requested = false;
  }
});

socket.on("rest_stops_data", data => {
  const list = document.getElementById("rest_stops");

  if (!list) return;

  list.innerHTML = "";

  if (!data.success) {
    list.innerHTML = `<li>Lỗi tìm điểm nghỉ: ${data.message || "Không rõ lỗi"}</li>`;
    return;
  }

  if (!data.stops || data.stops.length === 0) {
    list.innerHTML = "<li>Không tìm thấy điểm nghỉ gần đây.</li>";
    return;
  }

  clearPoiMarkers();

  data.stops.forEach(stop => {
    const li = document.createElement("li");

    li.innerHTML = `
      <b>${stop.name}</b><br>
      Loại: ${stop.type}<br>
      Cách khoảng: ${stop.distance_text}<br>
      <a href="${stop.map_url}" target="_blank">Mở trên OpenStreetMap</a>
      |
      <a href="${stop.direction_url}" target="_blank">Chỉ đường</a>
    `;

    list.appendChild(li);
    addPoiMarker(stop);
  });

  if (map && data.stops.length > 0) {
    const groupItems = poiMarkers.concat(userMarker ? [userMarker] : []);

    if (groupItems.length > 0) {
      const group = L.featureGroup(groupItems);
      map.fitBounds(group.getBounds().pad(0.2));
    }
  }
});