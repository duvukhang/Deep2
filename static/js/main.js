const socket = io();
let requested = false;

socket.on('update_data', (data) => {

    // ===== HIỂN THỊ CHỈ SỐ =====
    document.getElementById("ear_val").innerText = data.ear.toFixed(2);
    document.getElementById("pose_val").innerText = data.head_pose;
    document.getElementById("occlusion").innerText = data.eye_visible ? "Không" : "Có";

    // ===== TRẠNG THÁI =====
    if (data.is_drowsy) {
        document.getElementById("status").innerText = "NGỦ GẬT!";
        document.getElementById("status").className = "alert-box alert alert-danger text-center";

        // gọi map 1 lần
        if (!requested) {
            navigator.geolocation.getCurrentPosition((pos) => {
                socket.emit("find_stops_request", {
                    lat: pos.coords.latitude,
                    lon: pos.coords.longitude
                });
            });
            requested = true;
        }

    } else {
        document.getElementById("status").innerText = "TỈNH TÁO";
        document.getElementById("status").className = "alert-box alert alert-success text-center";
        requested = false;
    }
});

// ===== NHẬN DATA TRẠM NGHỈ =====
socket.on('rest_stops_data', (data) => {
    let list = document.getElementById("rest_stops");
    list.innerHTML = "";

    if (data.stops.length === 0) {
        list.innerHTML = "<li>Không tìm thấy</li>";
        return;
    }

    data.stops.forEach(s => {
        let li = document.createElement("li");
        li.innerText = s.name;
        list.appendChild(li);
    });
});