(() => {
  'use strict';

  const FPS = 50;
  const DT = 1.0 / FPS;
  const SCALE = 30.0;
  const MOTORS_TORQUE = 80;
  const SPEED_HIP = 4;
  const SPEED_KNEE = 6;
  const LIDAR_RANGE = 160 / SCALE;
  const INITIAL_RANDOM = 5;

  const HULL_POLY = [
    [-30, 9],
    [6, 9],
    [34, 1],
    [34, -8],
    [-30, -8]
  ];
  const LEG_DOWN = -8 / SCALE;
  const LEG_W = 8 / SCALE;
  const LEG_H = 34 / SCALE;

  const VIEWPORT_W = 600;
  const VIEWPORT_H = 400;

  const TERRAIN_STEP = 14 / SCALE;
  const TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4;
  const TERRAIN_STARTPAD = 20;
  const TERRAIN_GRASS = 10;
  const FRICTION = 2.5;

  const GRASS = 0, STUMP = 1, STAIRS = 2, PIT = 3, _STATES_ = 5;

  const canvas = document.getElementById('gameCanvas');
  const ctx = canvas.getContext('2d');
  const elDist = document.getElementById('hudDist');
  const elSpeed = document.getElementById('hudSpeed');
  const elScore = document.getElementById('hudScore');
  const elSteps = document.getElementById('hudSteps');
  const elRecord = document.getElementById('hudRecord');
  const elStatusDot = document.getElementById('statusDot');
  const elStatusText = document.getElementById('statusText');
  const btnPlayPause = document.getElementById('btnPlayPause');
  const btnReset = document.getElementById('btnReset');
  const btnSpeed1x = document.getElementById('btnSpeed1x');
  const btnSpeed2x = document.getElementById('btnSpeed2x');
  const btnSpeed5x = document.getElementById('btnSpeed5x');
  const btnFullscreen = document.getElementById('btnFullscreen');
  const chkLidar = document.getElementById('chkLidar');
  const chkHardcore = document.getElementById('chkHardcore');
  const chkAutoRestart = document.getElementById('chkAutoRestart');

  class PolicyNetwork {
    constructor(weights) {
      this.weights = weights;
      this.normObs = new Float32Array(24);
      this.h1 = new Float32Array(400);
      this.h2 = new Float32Array(300);
      this.action = new Float32Array(4);
    }

    forward(obs) {
      const w = this.weights;
      for (let i = 0; i < 24; i++) {
        const val = (obs[i] - w.obs_mean[i]) / Math.sqrt(w.obs_var[i] + 1e-6);
        this.normObs[i] = Math.max(-10.0, Math.min(10.0, val));
      }

      for (let j = 0; j < 400; j++) {
        let sum = w.d0_b[j];
        for (let i = 0; i < 24; i++) {
          sum += this.normObs[i] * w.d0_k[i][j];
        }
        this.h1[j] = sum;
      }
      let m1 = 0;
      for (let j = 0; j < 400; j++) m1 += this.h1[j];
      m1 /= 400;
      let v1 = 0;
      for (let j = 0; j < 400; j++) {
        const d = this.h1[j] - m1;
        v1 += d * d;
      }
      v1 /= 400;
      const invStd1 = 1.0 / Math.sqrt(v1 + 1e-6);
      for (let j = 0; j < 400; j++) {
        const norm = (this.h1[j] - m1) * invStd1 * w.ln0_scale[j] + w.ln0_bias[j];
        this.h1[j] = Math.max(0.0, norm);
      }

      for (let j = 0; j < 300; j++) {
        let sum = w.d1_b[j];
        for (let i = 0; i < 400; i++) {
          sum += this.h1[i] * w.d1_k[i][j];
        }
        this.h2[j] = sum;
      }
      let m2 = 0;
      for (let j = 0; j < 300; j++) m2 += this.h2[j];
      m2 /= 300;
      let v2 = 0;
      for (let j = 0; j < 300; j++) {
        const d = this.h2[j] - m2;
        v2 += d * d;
      }
      v2 /= 300;
      const invStd2 = 1.0 / Math.sqrt(v2 + 1e-6);
      for (let j = 0; j < 300; j++) {
        const norm = (this.h2[j] - m2) * invStd2 * w.ln1_scale[j] + w.ln1_bias[j];
        this.h2[j] = Math.max(0.0, norm);
      }

      for (let j = 0; j < 4; j++) {
        let sum = w.d2_b[j];
        for (let i = 0; i < 300; i++) {
          sum += this.h2[i] * w.d2_k[i][j];
        }
        this.action[j] = Math.tanh(sum);
      }
      return this.action;
    }
  }

  class Simulation {
    constructor() {
      this.Vec2 = planck.Vec2;
      this.policy = new PolicyNetwork(window.MODEL_WEIGHTS);

      this.running = true;
      this.speedMultiplier = 1;
      this.stepCount = 0;
      this.totalReward = 0;
      this.bestRecord = parseFloat(localStorage.getItem('bipedal_record') || '0');
      elRecord.textContent = this.bestRecord.toFixed(1) + ' m';

      this.gameOver = false;
      this.fallCooldown = 0;
      this.scrollX = 0;
      this.scrollY = TERRAIN_HEIGHT;

      this.initWorld();
      this.initEvents();
      this.resizeCanvas();
      window.addEventListener('resize', () => this.resizeCanvas());

      requestAnimationFrame(time => this.loop(time));
    }

    initWorld() {
      this.world = new planck.World(this.Vec2(0, -10));

      this.gameOver = false;
      this.fallCooldown = 0;
      this.stepCount = 0;
      this.totalReward = 0;
      this.prevShaping = null;
      this.leg0Contact = false;
      this.leg1Contact = false;

      this.world.on('begin-contact', contact => {
        const bA = contact.getFixtureA().getBody();
        const bB = contact.getFixtureB().getBody();
        if (bA === this.hull || bB === this.hull) {
          this.gameOver = true;
        }
        if (bA === this.legs[1] || bB === this.legs[1]) this.leg0Contact = true;
        if (bA === this.legs[3] || bB === this.legs[3]) this.leg1Contact = true;
      });

      this.world.on('end-contact', contact => {
        const bA = contact.getFixtureA().getBody();
        const bB = contact.getFixtureB().getBody();
        if (bA === this.legs[1] || bB === this.legs[1]) this.leg0Contact = false;
        if (bA === this.legs[3] || bB === this.legs[3]) this.leg1Contact = false;
      });

      this.terrainChunks = [];
      this.terrainState = GRASS;
      this.terrainVelocity = 0.0;
      this.terrainY = TERRAIN_HEIGHT;
      this.terrainCounter = TERRAIN_STARTPAD;
      this.terrainOneshot = false;
      this.stairSteps = 0;
      this.stairWidth = 0;
      this.stairHeight = 0;
      this.originalY = 0;
      this.lastTerrainStep = -1;
      this.lastTerrainX = 0;
      this.lastTerrainY = TERRAIN_HEIGHT;

      this.generateTerrainAhead(0, 100);

      const initX = TERRAIN_STEP * TERRAIN_STARTPAD / 2;
      const initY = TERRAIN_HEIGHT + 2 * LEG_H;

      this.hull = this.world.createDynamicBody(this.Vec2(initX, initY));
      this.hull.createFixture({
        shape: planck.Polygon(HULL_POLY.map(p => this.Vec2(p[0] / SCALE, p[1] / SCALE))),
        density: 5.0,
        friction: 0.1,
        filterCategoryBits: 0x0020,
        filterMaskBits: 0x0001,
        restitution: 0.0
      });
      this.hull.applyForceToCenter(
        this.Vec2((Math.random() * 2 - 1) * INITIAL_RANDOM, 0),
        true
      );

      this.legs = [];
      this.joints = [];

      for (const i of [-1, 1]) {
        const leg = this.world.createDynamicBody({
          position: this.Vec2(initX, initY - LEG_H / 2 - LEG_DOWN),
          angle: i * 0.05
        });
        leg.createFixture({
          shape: planck.Box(LEG_W / 2, LEG_H / 2),
          density: 1.0,
          restitution: 0.0,
          filterCategoryBits: 0x0020,
          filterMaskBits: 0x0001
        });
        const hipJoint = this.world.createJoint(new planck.RevoluteJoint({
          bodyA: this.hull,
          bodyB: leg,
          localAnchorA: this.Vec2(0, LEG_DOWN),
          localAnchorB: this.Vec2(0, LEG_H / 2),
          enableMotor: true,
          enableLimit: true,
          maxMotorTorque: MOTORS_TORQUE,
          motorSpeed: i,
          lowerAngle: -0.8,
          upperAngle: 1.1
        }));
        this.legs.push(leg);
        this.joints.push(hipJoint);

        const lower = this.world.createDynamicBody({
          position: this.Vec2(initX, initY - LEG_H * 3 / 2 - LEG_DOWN),
          angle: i * 0.05
        });
        lower.createFixture({
          shape: planck.Box(0.8 * LEG_W / 2, LEG_H / 2),
          density: 1.0,
          restitution: 0.0,
          filterCategoryBits: 0x0020,
          filterMaskBits: 0x0001
        });
        const kneeJoint = this.world.createJoint(new planck.RevoluteJoint({
          bodyA: leg,
          bodyB: lower,
          localAnchorA: this.Vec2(0, -LEG_H / 2),
          localAnchorB: this.Vec2(0, LEG_H / 2),
          enableMotor: true,
          enableLimit: true,
          maxMotorTorque: MOTORS_TORQUE,
          motorSpeed: 1,
          lowerAngle: -1.6,
          upperAngle: -0.1
        }));
        this.legs.push(lower);
        this.joints.push(kneeJoint);
      }

      this.lidarPoints = [];
      for (let i = 0; i < 10; i++) {
        this.lidarPoints.push({ p1: this.Vec2(0, 0), p2: this.Vec2(0, 0), frac: 1.0 });
      }

      this.clouds = [];
      for (let c = 0; c < 30; c++) {
        this.clouds.push({
          x: Math.random() * 300 - 30,
          y: (VIEWPORT_H / SCALE) * 0.75 + (Math.random() - 0.5) * 3,
          radius: 0.8 + Math.random() * 1.6
        });
      }

      this.scrollX = initX;
      this.scrollY = initY;
    }

    generateTerrainAhead(fromStep, toStep) {
      const isHardcore = chkHardcore.checked;

      for (let i = fromStep; i <= toStep; i++) {
        const x = i * TERRAIN_STEP;

        if (this.terrainState === GRASS && !this.terrainOneshot) {
          this.terrainVelocity = 0.8 * this.terrainVelocity + 0.01 * Math.sign(TERRAIN_HEIGHT - this.terrainY);
          if (i > TERRAIN_STARTPAD) {
            this.terrainVelocity += (Math.random() * 2 - 1) / SCALE;
          }
          this.terrainY += this.terrainVelocity;
        } else if (this.terrainState === PIT && this.terrainOneshot) {
          this.terrainCounter = 3 + Math.floor(Math.random() * 2);
          const b1 = this.world.createBody();
          b1.createFixture(planck.Polygon([
            this.Vec2(x, this.terrainY),
            this.Vec2(x + TERRAIN_STEP, this.terrainY),
            this.Vec2(x + TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP),
            this.Vec2(x, this.terrainY - 4 * TERRAIN_STEP)
          ]), { friction: FRICTION, filterCategoryBits: 0x0001 });

          const b2 = this.world.createBody();
          b2.createFixture(planck.Polygon([
            this.Vec2(x + this.terrainCounter * TERRAIN_STEP, this.terrainY),
            this.Vec2(x + (this.terrainCounter + 1) * TERRAIN_STEP, this.terrainY),
            this.Vec2(x + (this.terrainCounter + 1) * TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP),
            this.Vec2(x + this.terrainCounter * TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP)
          ]), { friction: FRICTION, filterCategoryBits: 0x0001 });

          this.terrainChunks.push({
            type: 'obstacle',
            bodies: [b1, b2],
            minX: x,
            maxX: x + (this.terrainCounter + 1) * TERRAIN_STEP,
            polys: [
              [[x, this.terrainY], [x + TERRAIN_STEP, this.terrainY], [x + TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP], [x, this.terrainY - 4 * TERRAIN_STEP]],
              [[x + this.terrainCounter * TERRAIN_STEP, this.terrainY], [x + (this.terrainCounter + 1) * TERRAIN_STEP, this.terrainY], [x + (this.terrainCounter + 1) * TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP], [x + this.terrainCounter * TERRAIN_STEP, this.terrainY - 4 * TERRAIN_STEP]]
            ]
          });

          this.terrainCounter += 2;
          this.originalY = this.terrainY;
        } else if (this.terrainState === PIT && !this.terrainOneshot) {
          this.terrainY = this.originalY;
          if (this.terrainCounter > 1) {
            this.terrainY -= 4 * TERRAIN_STEP;
          }
        } else if (this.terrainState === STUMP && this.terrainOneshot) {
          this.terrainCounter = 1 + Math.floor(Math.random() * 2);
          const b = this.world.createBody();
          const stumpPoly = [
            this.Vec2(x, this.terrainY),
            this.Vec2(x + this.terrainCounter * TERRAIN_STEP, this.terrainY),
            this.Vec2(x + this.terrainCounter * TERRAIN_STEP, this.terrainY + this.terrainCounter * TERRAIN_STEP),
            this.Vec2(x, this.terrainY + this.terrainCounter * TERRAIN_STEP)
          ];
          b.createFixture(planck.Polygon(stumpPoly), { friction: FRICTION, filterCategoryBits: 0x0001 });

          this.terrainChunks.push({
            type: 'stump',
            bodies: [b],
            minX: x,
            maxX: x + this.terrainCounter * TERRAIN_STEP,
            polys: [
              [[x, this.terrainY], [x + this.terrainCounter * TERRAIN_STEP, this.terrainY], [x + this.terrainCounter * TERRAIN_STEP, this.terrainY + this.terrainCounter * TERRAIN_STEP], [x, this.terrainY + this.terrainCounter * TERRAIN_STEP]]
            ]
          });
        } else if (this.terrainState === STAIRS && this.terrainOneshot) {
          this.stairHeight = Math.random() > 0.5 ? 1 : -1;
          this.stairWidth = 4;
          this.stairSteps = 3 + Math.floor(Math.random() * 2);
          this.originalY = this.terrainY;
          const stairBodies = [];
          const stairPolys = [];

          for (let s = 0; s < this.stairSteps; s++) {
            const b = this.world.createBody();
            const poly = [
              this.Vec2(x + (s * this.stairWidth) * TERRAIN_STEP, this.terrainY + (s * this.stairHeight) * TERRAIN_STEP),
              this.Vec2(x + ((1 + s) * this.stairWidth) * TERRAIN_STEP, this.terrainY + (s * this.stairHeight) * TERRAIN_STEP),
              this.Vec2(x + ((1 + s) * this.stairWidth) * TERRAIN_STEP, this.terrainY + (-1 + s * this.stairHeight) * TERRAIN_STEP),
              this.Vec2(x + (s * this.stairWidth) * TERRAIN_STEP, this.terrainY + (-1 + s * this.stairHeight) * TERRAIN_STEP)
            ];
            b.createFixture(planck.Polygon(poly), { friction: FRICTION, filterCategoryBits: 0x0001 });
            stairBodies.push(b);
            stairPolys.push(poly.map(p => [p.x, p.y]));
          }
          this.terrainChunks.push({
            type: 'stairs',
            bodies: stairBodies,
            minX: x,
            maxX: x + (this.stairSteps * this.stairWidth) * TERRAIN_STEP,
            polys: stairPolys
          });
          this.terrainCounter = this.stairSteps * this.stairWidth;
        } else if (this.terrainState === STAIRS && !this.terrainOneshot) {
          const s = this.stairSteps * this.stairWidth - this.terrainCounter - this.stairHeight;
          const n = s / this.stairWidth;
          this.terrainY = this.originalY + (n * this.stairHeight) * TERRAIN_STEP;
        }

        this.terrainOneshot = false;

        if (i > 0) {
          const edgeBody = this.world.createBody();
          edgeBody.createFixture(
            planck.Edge(this.Vec2(this.lastTerrainX, this.lastTerrainY), this.Vec2(x, this.terrainY)),
            { friction: FRICTION, filterCategoryBits: 0x0001 }
          );

          this.terrainChunks.push({
            type: 'ground',
            bodies: [edgeBody],
            minX: this.lastTerrainX,
            maxX: x,
            p1: [this.lastTerrainX, this.lastTerrainY],
            p2: [x, this.terrainY],
            stepIndex: i
          });
        }

        this.lastTerrainX = x;
        this.lastTerrainY = this.terrainY;
        this.lastTerrainStep = i;

        this.terrainCounter -= 1;
        if (this.terrainCounter <= 0) {
          this.terrainCounter = Math.floor(TERRAIN_GRASS / 2 + Math.random() * (TERRAIN_GRASS / 2));
          if (this.terrainState === GRASS && isHardcore) {
            this.terrainState = 1 + Math.floor(Math.random() * (_STATES_ - 1));
            this.terrainOneshot = true;
          } else {
            this.terrainState = GRASS;
            this.terrainOneshot = true;
          }
        }
      }
    }

    pruneTerrainBehind(minX) {
      let keepIndex = 0;
      for (let i = 0; i < this.terrainChunks.length; i++) {
        const chunk = this.terrainChunks[i];
        if (chunk.maxX < minX) {
          if (chunk.bodies) {
            for (const b of chunk.bodies) {
              this.world.destroyBody(b);
            }
          }
        } else {
          this.terrainChunks[keepIndex++] = chunk;
        }
      }
      this.terrainChunks.length = keepIndex;
    }

    stepSimulation() {
      if (!this.hull) return;

      const pos = this.hull.getPosition();
      const vel = this.hull.getLinearVelocity();
      const angVel = this.hull.getAngularVelocity();
      const angle = this.hull.getAngle();

      const currentStep = Math.floor(pos.x / TERRAIN_STEP);
      if (this.lastTerrainStep - currentStep < 100) {
        this.generateTerrainAhead(this.lastTerrainStep + 1, this.lastTerrainStep + 80);
      }
      this.pruneTerrainBehind(pos.x - 30);

      const lidarObs = new Float32Array(10);
      for (let i = 0; i < 10; i++) {
        let closestFrac = 1.0;
        const p1 = pos;
        const p2 = this.Vec2(
          pos.x + Math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
          pos.y - Math.cos(1.5 * i / 10.0) * LIDAR_RANGE
        );

        this.world.rayCast(p1, p2, (fixture, point, normal, fraction) => {
          if ((fixture.getFilterCategoryBits() & 1) === 0) return -1;
          if (fraction < closestFrac) closestFrac = fraction;
          return fraction;
        });

        lidarObs[i] = closestFrac;
        this.lidarPoints[i].p1 = p1;
        this.lidarPoints[i].p2 = p2;
        this.lidarPoints[i].frac = closestFrac;
      }

      const state = [
        angle,
        2.0 * angVel / FPS,
        0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
        0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        this.joints[0].getJointAngle(),
        this.joints[0].getJointSpeed() / SPEED_HIP,
        this.joints[1].getJointAngle() + 1.0,
        this.joints[1].getJointSpeed() / SPEED_KNEE,
        this.leg0Contact ? 1.0 : 0.0,
        this.joints[2].getJointAngle(),
        this.joints[2].getJointSpeed() / SPEED_HIP,
        this.joints[3].getJointAngle() + 1.0,
        this.joints[3].getJointSpeed() / SPEED_KNEE,
        this.leg1Contact ? 1.0 : 0.0,
        ...lidarObs
      ];

      const action = this.policy.forward(state);

      this.joints[0].setMotorSpeed(SPEED_HIP * Math.sign(action[0]));
      this.joints[0].setMaxMotorTorque(MOTORS_TORQUE * Math.max(0, Math.min(1, Math.abs(action[0]))));

      this.joints[1].setMotorSpeed(SPEED_KNEE * Math.sign(action[1]));
      this.joints[1].setMaxMotorTorque(MOTORS_TORQUE * Math.max(0, Math.min(1, Math.abs(action[1]))));

      this.joints[2].setMotorSpeed(SPEED_HIP * Math.sign(action[2]));
      this.joints[2].setMaxMotorTorque(MOTORS_TORQUE * Math.max(0, Math.min(1, Math.abs(action[2]))));

      this.joints[3].setMotorSpeed(SPEED_KNEE * Math.sign(action[3]));
      this.joints[3].setMaxMotorTorque(MOTORS_TORQUE * Math.max(0, Math.min(1, Math.abs(action[3]))));

      this.world.step(DT, 6 * 30, 2 * 30);
      this.world.clearForces();

      this.stepCount++;

      const shaping = (130 * pos.x / SCALE) - (5.0 * Math.abs(state[0]));
      let reward = 0;
      if (this.prevShaping !== null) {
        reward = shaping - this.prevShaping;
      }
      this.prevShaping = shaping;
      for (const a of action) {
        reward -= 0.00035 * MOTORS_TORQUE * Math.max(0, Math.min(1, Math.abs(a)));
      }
      this.totalReward += reward;

      const distance = Math.max(0, pos.x - (TERRAIN_STEP * TERRAIN_STARTPAD / 2));
      if (distance > this.bestRecord) {
        this.bestRecord = distance;
        localStorage.setItem('bipedal_record', this.bestRecord.toString());
        elRecord.textContent = this.bestRecord.toFixed(1) + ' m';
      }

      if (this.gameOver || pos.y < 0) {
        this.gameOver = true;
      }
    }

    loop(currentTime) {
      requestAnimationFrame(time => this.loop(time));

      if (this.running) {
        if (!this.gameOver) {
          for (let s = 0; s < this.speedMultiplier; s++) {
            this.stepSimulation();
            if (this.gameOver) break;
          }
        } else if (chkAutoRestart.checked) {
          this.fallCooldown++;
          if (this.fallCooldown > 45) {
            this.fallCooldown = 0;
            this.initWorld();
          }
        }
      }

      this.render();
      this.updateHUD();
    }

    render() {
      const w = canvas.width;
      const h = canvas.height;
      const dpr = window.devicePixelRatio || 1;
      ctx.clearRect(0, 0, w, h);

      if (!this.hull) return;

      const pos = this.hull.getPosition();
      this.scrollX += (pos.x - this.scrollX) * 0.1;
      this.scrollY += (pos.y - this.scrollY) * 0.1;

      const skyGrad = ctx.createLinearGradient(0, 0, 0, h);
      skyGrad.addColorStop(0, '#93c5fd');
      skyGrad.addColorStop(1, '#dbeafe');
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, w, h);

      ctx.save();
      ctx.translate(w * 0.35, h * 0.55);
      ctx.scale(dpr, -dpr);
      ctx.scale(SCALE, SCALE);
      ctx.translate(-this.scrollX, -this.scrollY);

      ctx.fillStyle = '#ffffff';
      for (const cloud of this.clouds) {
        const cx = cloud.x + this.scrollX * 0.2;
        const cy = cloud.y;
        ctx.beginPath();
        ctx.arc(cx, cy, cloud.radius, 0, Math.PI * 2);
        ctx.arc(cx + 0.6, cy + 0.1, cloud.radius * 0.8, 0, Math.PI * 2);
        ctx.arc(cx - 0.6, cy + 0.1, cloud.radius * 0.8, 0, Math.PI * 2);
        ctx.fill();
      }

      const bottomY = this.scrollY - 30;
      for (const chunk of this.terrainChunks) {
        if (chunk.type === 'ground') {
          const x1 = chunk.p1[0];
          const y1 = chunk.p1[1];
          const x2 = chunk.p2[0];
          const y2 = chunk.p2[1];

          ctx.fillStyle = '#6b5536';
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.lineTo(x2, bottomY);
          ctx.lineTo(x1, bottomY);
          ctx.closePath();
          ctx.fill();

          ctx.strokeStyle = chunk.stepIndex % 2 === 0 ? '#22c55e' : '#16a34a';
          ctx.lineWidth = 0.1;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        } else if (chunk.polys) {
          ctx.fillStyle = chunk.type === 'stump' ? '#854d0e' : '#cbd5e1';
          ctx.strokeStyle = '#475569';
          ctx.lineWidth = 0.05;

          for (const poly of chunk.polys) {
            ctx.beginPath();
            ctx.moveTo(poly[0][0], poly[0][1]);
            for (let p = 1; p < poly.length; p++) {
              ctx.lineTo(poly[p][0], poly[p][1]);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          }
        }
      }

      const flagX = TERRAIN_STEP * 3;
      const flagY1 = TERRAIN_HEIGHT;
      const flagY2 = flagY1 + 1.8;

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 0.08;
      ctx.beginPath();
      ctx.moveTo(flagX, flagY1);
      ctx.lineTo(flagX, flagY2);
      ctx.stroke();

      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.moveTo(flagX, flagY2);
      ctx.lineTo(flagX, flagY2 - 0.4);
      ctx.lineTo(flagX + 0.8, flagY2 - 0.2);
      ctx.closePath();
      ctx.fill();

      if (chkLidar.checked && !this.gameOver) {
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 0.04;
        for (const lp of this.lidarPoints) {
          const targetX = lp.p1.x + (lp.p2.x - lp.p1.x) * lp.frac;
          const targetY = lp.p1.y + (lp.p2.y - lp.p1.y) * lp.frac;

          ctx.beginPath();
          ctx.moveTo(lp.p1.x, lp.p1.y);
          ctx.lineTo(targetX, targetY);
          ctx.stroke();

          if (lp.frac < 0.99) {
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.arc(targetX, targetY, 0.08, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      for (let i = 0; i < this.legs.length; i++) {
        const body = this.legs[i];
        const trans = body.getTransform();
        const isUpper = (i === 0 || i === 2);
        const w = (isUpper ? LEG_W : 0.8 * LEG_W) / 2;
        const h = LEG_H / 2;

        const verts = [
          this.Vec2(-w, -h),
          this.Vec2(w, -h),
          this.Vec2(w, h),
          this.Vec2(-w, h)
        ].map(v => planck.Transform.mul(trans, v));

        ctx.fillStyle = (i < 2) ? '#818cf8' : '#a78bfa';
        ctx.strokeStyle = '#312e81';
        ctx.lineWidth = 0.05;

        ctx.beginPath();
        ctx.moveTo(verts[0].x, verts[0].y);
        for (let j = 1; j < verts.length; j++) {
          ctx.lineTo(verts[j].x, verts[j].y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      const hTrans = this.hull.getTransform();
      const hVerts = HULL_POLY.map(p =>
        planck.Transform.mul(hTrans, this.Vec2(p[0] / SCALE, p[1] / SCALE))
      );

      ctx.fillStyle = this.gameOver ? '#f87171' : '#6366f1';
      ctx.strokeStyle = '#3730a3';
      ctx.lineWidth = 0.06;

      ctx.beginPath();
      ctx.moveTo(hVerts[0].x, hVerts[0].y);
      for (let j = 1; j < hVerts.length; j++) {
        ctx.lineTo(hVerts[j].x, hVerts[j].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      const eyePos = planck.Transform.mul(hTrans, this.Vec2(24 / SCALE, 2 / SCALE));
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(eyePos.x, eyePos.y, 0.12, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    }

    updateHUD() {
      if (!this.hull) return;
      const pos = this.hull.getPosition();
      const vel = this.hull.getLinearVelocity();
      const dist = Math.max(0, pos.x - (TERRAIN_STEP * TERRAIN_STARTPAD / 2));

      elDist.textContent = dist.toFixed(1) + ' m';
      elSpeed.textContent = Math.max(0, vel.x).toFixed(1) + ' m/s';
      elScore.textContent = this.totalReward.toFixed(1);
      elSteps.textContent = this.stepCount.toLocaleString();

      if (this.gameOver) {
        elStatusDot.classList.add('fallen');
        elStatusText.textContent = chkAutoRestart.checked ? 'Restarting' : 'Fallen';
      } else {
        elStatusDot.classList.remove('fallen');
        elStatusText.textContent = this.running ? 'Walking' : 'Paused';
      }
    }

    resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
    }

    initEvents() {
      btnPlayPause.addEventListener('click', () => {
        this.running = !this.running;
        btnPlayPause.textContent = this.running ? '⏸ Pause' : '▶ Play';
        if (this.running) btnPlayPause.classList.add('primary');
        else btnPlayPause.classList.remove('primary');
      });

      btnReset.addEventListener('click', () => {
        this.initWorld();
      });

      const setSpeed = (mul, activeBtn) => {
        this.speedMultiplier = mul;
        [btnSpeed1x, btnSpeed2x, btnSpeed5x].forEach(b => b.classList.remove('active'));
        activeBtn.classList.add('active');
      };

      btnSpeed1x.addEventListener('click', () => setSpeed(1, btnSpeed1x));
      btnSpeed2x.addEventListener('click', () => setSpeed(2, btnSpeed2x));
      btnSpeed5x.addEventListener('click', () => setSpeed(5, btnSpeed5x));

      btnFullscreen.addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      });

      chkHardcore.addEventListener('change', () => {
        this.initWorld();
      });

      window.addEventListener('keydown', e => {
        if (e.code === 'Space') {
          e.preventDefault();
          btnPlayPause.click();
        } else if (e.code === 'KeyR') {
          btnReset.click();
        } else if (e.code === 'KeyL') {
          chkLidar.checked = !chkLidar.checked;
        } else if (e.code === 'KeyF') {
          btnFullscreen.click();
        }
      });
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    if (!window.MODEL_WEIGHTS) return;
    new Simulation();
  });
})();
