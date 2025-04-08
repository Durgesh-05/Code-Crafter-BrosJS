'use client';

import { io } from 'socket.io-client';

export const socket = io('http://localhost:8000', {
  reconnection: true,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
  transports: ['websocket'],
});
