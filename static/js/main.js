const socket = io();

let requested = false;
let map = null;
let userMarker = null;
let destinationMarker = null;
let routeLayer = null;
let destinationRouteLayer = null;
let poiMarkers = [];
let lastDrowsyTime = 0;
let lastKnownLocation = null;
let pendingRouteRequest = null;

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setRouteStatus(value) {
  setText("route_status", value);
}

function setStatusBox(state, isDrowsy) {
  const status = document.getElementById("status");
  if (!status) return;

  if (state === "DROWSY_CONFIRMED" || isDrowsy) {
    status.innerText = "NGỦ GẬT!";
    status.className = "alert-box alert alert-danger text-center";
  } else if (state === "WARNING_SUNGLASSES_MODE") {
    status.innerText = "CẢNH BÁO - KÍNH RÂM";
    status.className = "alert-box alert alert-warning text-center";
  } else if (state === "WARNING_MASK_MODE") {
    status.innerText = "CẢNH BÁO - KHẨU TRANG";
    status.className = "alert-box alert alert-warning text-center";
  } else if (state === "WARNING_POSTURE_MODE") {
    status.innerText = "CẢNH BÁO TƯ THẾ NGỦ GẬT";
    status.className = "alert-box alert alert-warning text-center";
  } else if (state === "SUNGLASSES_MODE") {
    status.innerText = "CHẾ ĐỘ KÍNH RÂM";
    status.className = "alert-box alert alert-info text-center";
  } else if (state === "MASK_MODE") {
    status.innerText = "CHẾ ĐỘ KHẨU TRANG";
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

function modeLabel(mode) {
  const labels = {
    FULL_FACE_MODE: "Mắt + miệng + tư thế",
    SUNGLASSES_MOUTH_POSE_MODE: "Kính râm: miệng + tư thế",
    MASK_EYE_POSE_MODE: "Khẩu trang: mắt + tư thế",
    EYE_POSE_MODE: "Mắt + tư thế",
    MOUTH_POSE_MODE: "Miệng + tư thế",
    CAMERA_BAD_MODE: "Camera/góc kém",
    NO_FACE_MODE: "Không thấy mặt"
  };

  return labels[mode] || mode || "FULL_FACE_MODE";
}

function lightingLabel(mode) {
  const labels = {
    NORMAL: "Bình thường",
    LOW_LIGHT: "Tăng sáng",
    GLARE: "Giảm chói",
    LOW_CONTRAST: "Tăng tương phản"
  };

  return labels[mode] || mode || "Bình thường";
}

function postureLabel(status) {
  const labels = {
    STABLE: "Ổn định",
    HEAD_NOD: "Vừa gật đầu",
    HEAD_NOD_WARNING: "Gật đầu lặp lại",
    HEAD_NOD_REPEAT: "Gật đầu nhiều lần",
    LEAN_REPEAT: "Đổ/chồm người",
    LEAN_WARNING: "Đổ/chồm lặp lại"
  };

  return labels[status] || status || "Ổn định";
}

function occlusionLabel(data) {
  if (data.mask_detected) return "Khẩu trang";
  if (data.sunglasses_detected) return "Kính râm";
  if (!data.eye_visible && !data.mouth_visible) return "Mắt và miệng bị che";
  if (!data.eye_visible) return "Mắt bị che";
  if (!data.mouth_visible) return "Miệng bị che";
  return "Không";
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

function updateDestinationMarker(destination) {
  if (!map || !destination) return;

  if (destinationMarker) {
    map.removeLayer(destinationMarker);
  }

  destinationMarker = L.marker([destination.lat, destination.lon])
    .addTo(map)
    .bindPopup(`<b>${escapeHtml(destination.name)}</b><br>${escapeHtml(destination.display_name)}`);
}

function clearRouteLayers() {
  if (!map) return;

  if (routeLayer) {
    map.removeLayer(routeLayer);
    routeLayer = null;
  }

  if (destinationRouteLayer) {
    map.removeLayer(destinationRouteLayer);
    destinationRouteLayer = null;
  }
}

function drawRouteLayer(route, options = {}) {
  if (!map || !route || !route.geometry || route.geometry.length < 2) return null;

  const layer = L.polyline(route.geometry, {
    color: options.color || "#38bdf8",
    weight: options.weight || 5,
    opacity: options.opacity || 0.9,
    dashArray: options.dashArray || null
  }).addTo(map);

  return layer;
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

  const routeInfo = stop.route_text ? `<br>${stop.route_text}` : "";
  const encodedName = encodeURIComponent(stop.name || "điểm nghỉ");
  const popupHtml = `
    <b>${escapeHtml(stop.name)}</b><br>
    ${escapeHtml(stop.type)}<br>
    Cách khoảng ${stop.distance_text}${routeInfo}<br>
    <a href="${stop.direction_url}" target="_blank">Chỉ đường</a>
    <button
      class="popup-route-btn"
      onclick="routeToStop(${Number(stop.lat)}, ${Number(stop.lon)}, decodeURIComponent('${encodedName}'))"
    >
      Dẫn tới đây
    </button>
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
          heading: Number.isFinite(position.coords.heading)
            ? position.coords.heading
            : null,
          speed: Number.isFinite(position.coords.speed)
            ? position.coords.speed
            : null,
          source: "GPS trình duyệt"
        });
      },
      error => reject(error),
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
    heading: null,
    speed: null,
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

function bearingBetween(lat1, lon1, lat2, lon2) {
  const toRad = value => value * Math.PI / 180;
  const toDeg = value => value * 180 / Math.PI;
  const lat1Rad = toRad(lat1);
  const lat2Rad = toRad(lat2);
  const deltaLon = toRad(lon2 - lon1);
  const y = Math.sin(deltaLon) * Math.cos(lat2Rad);
  const x = (
    Math.cos(lat1Rad) * Math.sin(lat2Rad)
    - Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(deltaLon)
  );

  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

function findNearestStop(stops) {
  if (!stops || stops.length === 0) return null;

  return stops.reduce((nearest, stop) => {
    if (!nearest) return stop;
    return Number(stop.distance_km) < Number(nearest.distance_km) ? stop : nearest;
  }, null);
}

async function fetchRoute(start, end) {
  const params = new URLSearchParams({
    start_lat: start.lat,
    start_lon: start.lon,
    end_lat: end.lat,
    end_lon: end.lon
  });
  const res = await fetch(`/api/route?${params.toString()}`);
  const data = await res.json();

  if (!data.success) {
    throw new Error(data.message || "Không vẽ được tuyến đường");
  }

  return data.route;
}

async function drawRouteToStop(stop, destination = null) {
  if (!lastKnownLocation || !stop) return;

  clearRouteLayers();
  setRouteStatus(`Đang vẽ đường tới ${stop.name}...`);

  const start = {
    lat: lastKnownLocation.lat,
    lon: lastKnownLocation.lon
  };
  const stopPoint = {
    lat: stop.lat,
    lon: stop.lon
  };

  const route = await fetchRoute(start, stopPoint);
  routeLayer = drawRouteLayer(route, {
    color: "#38bdf8",
    weight: 6
  });

  let status = `Đường tới ${stop.name}: ${route.distance_km} km`;

  if (route.duration_min !== null) {
    status += `, khoảng ${route.duration_min} phút`;
  }

  if (destination) {
    const destinationRoute = await fetchRoute(stopPoint, destination);
    destinationRouteLayer = drawRouteLayer(destinationRoute, {
      color: "#94a3b8",
      weight: 4,
      opacity: 0.72,
      dashArray: "8 8"
    });
  }

  const fitItems = [];
  if (routeLayer) fitItems.push(routeLayer);
  if (destinationRouteLayer) fitItems.push(destinationRouteLayer);
  if (userMarker) fitItems.push(userMarker);
  if (destinationMarker) fitItems.push(destinationMarker);

  if (fitItems.length > 0) {
    const group = L.featureGroup(fitItems);
    map.fitBounds(group.getBounds().pad(0.18));
  }

  setRouteStatus(status);
}

async function routeToStop(lat, lon, name) {
  try {
    if (!lastKnownLocation) {
      const coords = await getBestLocation();
      lastKnownLocation = {
        lat: coords.latitude,
        lon: coords.longitude
      };
      initMap(lastKnownLocation.lat, lastKnownLocation.lon);
      updateUserMarker(lastKnownLocation.lat, lastKnownLocation.lon);
    }

    await drawRouteToStop({
      lat,
      lon,
      name: name || "điểm nghỉ",
      distance_km: 0
    });
  } catch (err) {
    console.error("Lỗi vẽ đường:", err);
    setRouteStatus("Không vẽ được đường tới điểm nghỉ này.");
  }
}

async function requestNearbyStops(options = {}) {
  try {
    setText("location_text", "Đang lấy vị trí...");

    const radiusSelect = document.getElementById("radius_select");
    const radius = radiusSelect ? parseInt(radiusSelect.value, 10) : 60000;

    const coords = await getBestLocation();
    const lat = coords.latitude;
    const lon = coords.longitude;
    const destination = options.destination || null;
    const heading = destination
      ? bearingBetween(lat, lon, destination.lat, destination.lon)
      : coords.heading;

    lastKnownLocation = {
      lat,
      lon,
      heading,
      source: coords.source
    };
    pendingRouteRequest = options.routeToNearest
      ? {
          destination,
          routeToNearest: true
        }
      : null;

    initMap(lat, lon);

    const label = coords.city
      ? `Vị trí theo ${coords.source}: ${coords.city}, ${coords.country}`
      : `Vị trí theo ${coords.source}`;

    updateUserMarker(lat, lon, label);

    if (destination) {
      updateDestinationMarker(destination);
      setRouteStatus(`Đã tìm thấy: ${destination.display_name}`);
    } else {
      setRouteStatus("");
      clearRouteLayers();
    }

    setText(
      "location_text",
      `${label} | lat: ${lat.toFixed(5)}, lon: ${lon.toFixed(5)}`
    );

    socket.emit("find_stops_request", {
      lat,
      lon,
      radius,
      heading,
      speed: coords.speed
    });
  } catch (err) {
    console.error("Lỗi lấy vị trí:", err);
    setText(
      "location_text",
      "Không lấy được vị trí. Hãy bật quyền vị trí hoặc kiểm tra mạng."
    );
    setRouteStatus("Không lấy được vị trí hiện tại.");
  }
}

async function searchAddressAndRoute() {
  const input = document.getElementById("address_input");
  const query = input ? input.value.trim() : "";

  if (!query) {
    setRouteStatus("Nhập địa chỉ cần tìm đường.");
    return;
  }

  try {
    setRouteStatus("Đang tìm địa chỉ...");
    const params = new URLSearchParams({ q: query });
    const res = await fetch(`/api/geocode?${params.toString()}`);
    const data = await res.json();

    if (!data.success || !data.results || data.results.length === 0) {
      setRouteStatus("Không tìm thấy địa chỉ phù hợp.");
      return;
    }

    const destination = data.results[0];
    await requestNearbyStops({
      destination,
      routeToNearest: true
    });
  } catch (err) {
    console.error("Lỗi tìm đường:", err);
    setRouteStatus("Không tìm đường được. Kiểm tra mạng hoặc địa chỉ.");
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
  const headPitchDelta = Number(data.head_pitch_delta || 0);
  const headYawDelta = Number(data.head_yaw_delta || 0);
  const headRollDelta = Number(data.head_roll_delta || 0);
  const brightness = Number(data.brightness || 0);
  const contrast = Number(data.contrast || 0);
  const postureScore = Number(data.posture_score || 0);

  setText("ear_val", ear.toFixed(2));
  setText("threshold_val", threshold.toFixed(2));
  setText("mar_val", mar.toFixed(2));
  setText("pose_val", pose.toFixed(2));
  setText(
    "head_angles_val",
    `${headPitchDelta.toFixed(0)}° / ${headYawDelta.toFixed(0)}° / ${headRollDelta.toFixed(0)}°`
  );
  setText("score_val", score.toFixed(2));
  setText("state_val", data.state || "NORMAL");
  setText("face_count", data.face_count ?? 0);
  setText("visible_eye_count", data.visible_eye_count ?? 0);
  setText("confidence_val", confidence.toFixed(2));
  setText("lighting_mode_val", lightingLabel(data.lighting_mode));
  setText("brightness_val", `${brightness.toFixed(0)} / ${contrast.toFixed(0)}`);
  setText("posture_status_val", postureLabel(data.posture_status));
  setText("nod_count_val", data.nod_count ?? 0);
  setText("lean_event_count_val", data.lean_event_count ?? 0);
  setText("posture_score_val", postureScore.toFixed(2));
  setText("leaning_val", data.is_leaning ? "Có" : "Không");
  setText("occlusion", occlusionLabel(data));
  setText("detection_mode_val", modeLabel(data.detection_mode));
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

  if (
    data.state === "NORMAL"
    || data.state === "SUNGLASSES_MODE"
    || data.state === "MASK_MODE"
  ) {
    requested = false;
  }
});

socket.on("rest_stops_data", async data => {
  const list = document.getElementById("rest_stops");
  if (!list) return;

  list.innerHTML = "";

  if (!data.success) {
    list.innerHTML = `<li>Lỗi tìm điểm nghỉ: ${data.message || "Không rõ lỗi"}</li>`;
    return;
  }

  if (!data.stops || data.stops.length === 0) {
    list.innerHTML = "<li>Không tìm thấy điểm nghỉ gần đây.</li>";
    setRouteStatus("Không tìm thấy điểm nghỉ để dẫn đường.");
    return;
  }

  clearPoiMarkers();

  data.stops.forEach(stop => {
    const li = document.createElement("li");
    const routeInfo = stop.route_text
      ? `<br><span>${stop.route_text}</span>`
      : "";
    const encodedName = encodeURIComponent(stop.name || "điểm nghỉ");

    li.innerHTML = `
      <b>${escapeHtml(stop.name)}</b><br>
      Loại: ${escapeHtml(stop.type)}<br>
      Cách khoảng: ${stop.distance_text}${routeInfo}<br>
      <a href="${stop.map_url}" target="_blank">Mở trên OpenStreetMap</a>
      |
      <a href="${stop.direction_url}" target="_blank">Chỉ đường</a>
      |
      <button
        class="inline-route-btn"
        onclick="routeToStop(${Number(stop.lat)}, ${Number(stop.lon)}, decodeURIComponent('${encodedName}'))"
      >
        Dẫn tới đây
      </button>
    `;

    list.appendChild(li);
    addPoiMarker(stop);
  });

  if (pendingRouteRequest && pendingRouteRequest.routeToNearest) {
    const nearestStop = data.nearest_stop || findNearestStop(data.stops);

    if (nearestStop) {
      try {
        await drawRouteToStop(nearestStop, pendingRouteRequest.destination);
      } catch (err) {
        console.error("Lỗi vẽ đường tới điểm nghỉ gần nhất:", err);
        setRouteStatus("Đã tìm được điểm nghỉ nhưng chưa vẽ được tuyến đường.");
      }
    }
  }

  if (map && data.stops.length > 0) {
    const groupItems = poiMarkers
      .concat(userMarker ? [userMarker] : [])
      .concat(destinationMarker ? [destinationMarker] : [])
      .concat(routeLayer ? [routeLayer] : [])
      .concat(destinationRouteLayer ? [destinationRouteLayer] : []);

    if (groupItems.length > 0) {
      const group = L.featureGroup(groupItems);
      map.fitBounds(group.getBounds().pad(0.2));
    }
  }
});

async function loadAlertHistory() {
  try {
    const res = await fetch("/api/alerts");
    const data = await res.json();

    if (!data.success) return;

    setText("total_alerts", data.stats.total_alerts);
    setText("drowsy_count", data.stats.drowsy_count);
    setText("warning_count", data.stats.warning_count);

    const list = document.getElementById("alert_history");
    if (!list) return;

    list.innerHTML = "";

    if (!data.logs || data.logs.length === 0) {
      list.innerHTML = "<li>Chưa có cảnh báo</li>";
      return;
    }

    data.logs.forEach(item => {
      const li = document.createElement("li");

      li.innerHTML = `
        <b>${item.time}</b><br>
        Trạng thái: ${item.state}<br>
        EAR: ${item.ear} | MAR: ${item.mar}<br>
        Score: ${item.drowsy_score}
      `;

      list.appendChild(li);
    });
  } catch (err) {
    console.error("Không tải được lịch sử cảnh báo:", err);
  }
}

setInterval(loadAlertHistory, 5000);
loadAlertHistory();
