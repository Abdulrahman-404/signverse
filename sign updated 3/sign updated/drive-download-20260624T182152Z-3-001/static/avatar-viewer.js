// ============================================================
//  Sign Language Avatar viewer — main-app integration
//  Adapted from Avatar/static/js/avatar.js: same Three.js scene
//  setup, glTF bone retargeting rig, and SiGML parsing, but
//  driven by the current translation's SiGML instead of a static
//  demo file, with no manual scrubber/select UI — it auto-plays
//  every sign in the result once, then returns to an idle pose.
// ============================================================

(function () {
'use strict';

const canvas   = document.getElementById('avatar-canvas');
const statusEl = document.getElementById('avatar-status');

if (!canvas) return; // page doesn't have the avatar widget

// ─── Renderer / scene / camera ────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled   = true;
renderer.shadowMap.type      = THREE.PCFSoftShadowMap;
renderer.outputEncoding      = THREE.sRGBEncoding;
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11142b);

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
camera.position.set(0, 1.35, 3.4);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.dampingFactor  = 0.08;
controls.minDistance    = 1.4;
controls.maxDistance    = 6;
controls.maxPolarAngle  = Math.PI * 0.55;
controls.update();

function resize() {
    const parent = canvas.parentElement;
    const w = parent.clientWidth  || 420;
    const h = parent.clientHeight || 420;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(canvas.parentElement);

// ─── Lighting ──────────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.05);
key.position.set(2, 4, 3);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.radius = 3;
key.shadow.bias   = -0.0004;
scene.add(key);
const fillLight = new THREE.DirectionalLight(0xcfe0ff, 0.32);
fillLight.position.set(-3, 2, -2);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xdfe8ff, 0.5);
rimLight.position.set(0, 3, -4);
scene.add(rimLight);

const ground = new THREE.Mesh(
    new THREE.CircleGeometry(4, 48),
    new THREE.MeshStandardMaterial({ color: 0x8a97ab, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const aoBlob = new THREE.Mesh(
    new THREE.CircleGeometry(0.45, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22 })
);
aoBlob.rotation.x = -Math.PI / 2;
aoBlob.position.y = 0.001;
scene.add(aoBlob);

// ─── Idle hand wiggle (plays whenever no sign is active) ────────
function captureBase(bone) {
    return { bone, baseQuat: bone.quaternion.clone() };
}
const handRig = { wrists: [], fingers: [] };
function buildIdleHandRig(boneMap) {
    if (boneMap.LeftHand)  handRig.wrists.push({ ...captureBase(boneMap.LeftHand),  side: -1 });
    if (boneMap.RightHand) handRig.wrists.push({ ...captureBase(boneMap.RightHand), side:  1 });
    Object.keys(boneMap).forEach((name) => {
        if (/^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)[1-3]$/.test(name)) {
            handRig.fingers.push({ ...captureBase(boneMap[name]) });
        }
    });
}
function animateIdleHands(t) {
    handRig.wrists.forEach(({ bone, baseQuat, side }) => {
        const wobble = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            Math.sin(t * 0.5) * 0.03, 0, Math.sin(t * 0.4) * 0.02 * side
        ));
        bone.quaternion.copy(baseQuat).multiply(wobble);
    });
    const baseCurl = Math.sin(t * 0.7) * 0.05;
    handRig.fingers.forEach(({ bone, baseQuat }) => {
        const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, baseCurl * 0.85));
        bone.quaternion.copy(baseQuat).multiply(q);
    });
}

// ─── SiGML parsing ───────────────────────────────────────────────
function parsePoints(element) {
    const pts = {};
    element.querySelectorAll('point').forEach((p) => {
        pts[+p.getAttribute('id')] = {
            x: +p.getAttribute('x'), y: +p.getAttribute('y'), z: +p.getAttribute('z'),
        };
    });
    const arr = [];
    const len = Object.keys(pts).length;
    for (let i = 0; i < len; i++) arr.push(pts[i] || { x: 0, y: 0, z: 0 });
    return arr;
}

function parseSigml(xmlStr) {
    const doc = new DOMParser().parseFromString(xmlStr, 'application/xml');
    const signs = [];
    doc.querySelectorAll('hamgestural_sign').forEach((sign) => {
        const gloss       = sign.getAttribute('gloss') || '';
        const arabicGloss = sign.getAttribute('arabic_gloss') || gloss;
        const fps         = +(sign.getAttribute('fps') || 30);
        const durationMs  = +(sign.getAttribute('duration_ms') || 1000);
        const frames = [];
        sign.querySelectorAll('frame').forEach((fr) => {
            const poseEl = fr.querySelector('pose');
            const lhEl   = fr.querySelector('left_hand');
            const rhEl   = fr.querySelector('right_hand');
            frames.push({
                pose:       poseEl ? parsePoints(poseEl) : null,
                left_hand:  lhEl   ? parsePoints(lhEl)  : null,
                right_hand: rhEl   ? parsePoints(rhEl)  : null,
            });
        });
        if (frames.length) signs.push({ gloss, arabicGloss, fps, durationMs, frames });
    });
    return signs;
}

// ─── MediaPipe landmark indices ──────────────────────────────────
const MP = { L_SHOULDER: 11, R_SHOULDER: 12, L_ELBOW: 13, R_ELBOW: 14, L_WRIST: 15, R_WRIST: 16 };
const H = {
    WRIST: 0,
    THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
    INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
    MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
    RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
    PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};
const FINGER_LM = {
    Thumb:  ['THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP'],
    Index:  ['INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP'],
    Middle: ['MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP'],
    Ring:   ['RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP'],
    Pinky:  ['PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'],
};

function mpDir(a, b) {
    return new THREE.Vector3(-(b.x - a.x), -(b.y - a.y), -(b.z - a.z));
}

// ─── Bone retargeting rig ─────────────────────────────────────────
const RIG = [];

function addChain(boneMap, boneName, childName, targetFn) {
    const bone  = boneMap[boneName];
    const child = boneMap[childName];
    if (!bone || !child) return;

    const boneWorldPos  = new THREE.Vector3(); bone.getWorldPosition(boneWorldPos);
    const childWorldPos = new THREE.Vector3(); child.getWorldPosition(childWorldPos);
    const boneWorldQuat = new THREE.Quaternion(); bone.getWorldQuaternion(boneWorldQuat);

    const worldDir = childWorldPos.clone().sub(boneWorldPos);
    if (worldDir.lengthSq() < 1e-10) return;
    worldDir.normalize();
    const localDir = worldDir.applyQuaternion(boneWorldQuat.clone().invert()).normalize();

    RIG.push({ bone, localDir, targetFn });
}

function buildRetargetRig(boneMap) {
    addChain(boneMap, 'LeftArm', 'LeftForeArm', (f) => f.pose && mpDir(f.pose[MP.R_SHOULDER], f.pose[MP.R_ELBOW]));
    addChain(boneMap, 'LeftForeArm', 'LeftHand', (f) => f.pose && mpDir(f.pose[MP.R_ELBOW], f.pose[MP.R_WRIST]));
    addChain(boneMap, 'RightArm', 'RightForeArm', (f) => f.pose && mpDir(f.pose[MP.L_SHOULDER], f.pose[MP.L_ELBOW]));
    addChain(boneMap, 'RightForeArm', 'RightHand', (f) => f.pose && mpDir(f.pose[MP.L_ELBOW], f.pose[MP.L_WRIST]));

    addChain(boneMap, 'LeftHand', 'LeftHandMiddle1', (f) => f.right_hand && mpDir(f.right_hand[H.WRIST], f.right_hand[H.MIDDLE_MCP]));
    addChain(boneMap, 'RightHand', 'RightHandMiddle1', (f) => f.left_hand && mpDir(f.left_hand[H.WRIST], f.left_hand[H.MIDDLE_MCP]));

    [['Left', 'right_hand'], ['Right', 'left_hand']].forEach(([side, key]) => {
        Object.entries(FINGER_LM).forEach(([finger, lm]) => {
            for (let j = 1; j <= 3; j++) {
                const idA = H[lm[j - 1]], idB = H[lm[j]];
                addChain(boneMap, `${side}Hand${finger}${j}`, `${side}Hand${finger}${j + 1}`,
                    (f) => f[key] && mpDir(f[key][idA], f[key][idB]));
            }
        });
    });
}

function applyRetarget(frame) {
    RIG.forEach(({ bone, localDir, targetFn }) => {
        const targetWorldDir = targetFn(frame);
        if (!targetWorldDir || targetWorldDir.lengthSq() < 1e-8) return;
        targetWorldDir.normalize();

        const parentWorldQuat = new THREE.Quaternion();
        bone.parent.getWorldQuaternion(parentWorldQuat);
        const targetLocalDir = targetWorldDir.applyQuaternion(parentWorldQuat.clone().invert());

        bone.quaternion.setFromUnitVectors(localDir, targetLocalDir);
        bone.updateMatrixWorld(true);
    });
}

// ─── Playback: auto-play every sign in the result once ────────────
let signQueue     = [];
let queueIdx       = 0;
let activeSign     = null;
let frameIdx       = 0;
let playing        = false;
let lastTime       = 0;
let msPerFrame     = 33;
let usingSignPose  = false;
let modelReady     = false;
let pendingSigmlXml = null;

function setStatus(txt, isPlaying) {
    if (!statusEl) return;
    statusEl.textContent = txt;
    statusEl.className   = 'avatar-status' + (isPlaying ? ' playing' : '');
}

function setActiveSign(sign) {
    activeSign = sign;
    frameIdx   = 0;
    msPerFrame = 1000 / (sign.fps || 30);
    applyRetarget(sign.frames[0]);
    usingSignPose = true;
    setStatus(sign.arabicGloss || sign.gloss, true);
}

function playSequence(signs) {
    signQueue = signs || [];
    queueIdx  = 0;
    if (!signQueue.length) {
        setStatus('لا توجد إشارات لعرضها', false);
        return;
    }
    setActiveSign(signQueue[0]);
    playing  = true;
    lastTime = performance.now();
}

function advanceQueue() {
    queueIdx += 1;
    if (queueIdx >= signQueue.length) {
        playing = false;
        usingSignPose = false;
        setStatus('انتهى العرض', false);
        return;
    }
    setActiveSign(signQueue[queueIdx]);
}

// ─── Load avatar model ─────────────────────────────────────────────
const clock = new THREE.Clock();
const loader = new THREE.GLTFLoader();
setStatus('جارِ تحميل المجسم ثلاثي الأبعاد…', false);

loader.load(
    'static/models/avatar.glb',
    (gltf) => {
        const model = gltf.scene;
        // Baked-in 100x FBX-centimeter scale artifact from the Sketchfab export.
        model.scale.setScalar(0.01);
        model.traverse((obj) => {
            if (obj.isMesh) {
                obj.castShadow    = true;
                obj.receiveShadow = true;
                const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                mats.forEach((mat) => {
                    if (mat.map) mat.map.encoding = THREE.sRGBEncoding;
                    mat.needsUpdate = true;
                });
            }
        });
        scene.add(model);
        model.updateMatrixWorld(true);

        const boneMap = {};
        model.traverse((o) => { boneMap[o.name.replace(/_[0-9]+$/, '')] = o; });

        buildIdleHandRig(boneMap);
        buildRetargetRig(boneMap);

        modelReady = true;
        setStatus('جاهز', false);

        if (pendingSigmlXml) {
            const signs = parseSigml(pendingSigmlXml);
            pendingSigmlXml = null;
            playSequence(signs);
        }
    },
    undefined,
    (err) => {
        console.error(err);
        setStatus('تعذر تحميل المجسم ثلاثي الأبعاد', false);
    }
);

// ─── Render loop ───────────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    clock.getDelta();

    if (playing && activeSign) {
        const dt = now - lastTime;
        if (dt >= msPerFrame) {
            lastTime = now;
            frameIdx++;
            if (frameIdx >= activeSign.frames.length) {
                advanceQueue();
            } else {
                applyRetarget(activeSign.frames[frameIdx]);
            }
        }
    } else if (!usingSignPose) {
        animateIdleHands(clock.getElapsedTime());
    }

    controls.update();
    renderer.render(scene, camera);
}
animate();

// ─── Public API ────────────────────────────────────────────────
window.AvatarViewer = {
    playSigml(xmlText) {
        const signs = parseSigml(xmlText);
        if (!modelReady) {
            pendingSigmlXml = xmlText;
            setStatus('جارِ تحميل المجسم ثلاثي الأبعاد…', false);
            return;
        }
        playSequence(signs);
    },
};

})();
