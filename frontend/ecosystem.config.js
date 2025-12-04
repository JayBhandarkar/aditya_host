module.exports = {
  apps: [{
    name: 'aditya-frontend',
    script: 'npm',
    args: 'start',
    cwd: './',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
};