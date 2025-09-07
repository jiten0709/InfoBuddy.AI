# Build stage
FROM node:20-bookworm-slim AS builder
WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm ci

# Set build-time env for public runtime (embedded in bundle)
ARG NEXT_PUBLIC_SERVER_URL
ENV NEXT_PUBLIC_SERVER_URL=${NEXT_PUBLIC_SERVER_URL}

# Copy source and build
COPY . .
RUN npm run build

# Runtime stage
FROM node:20-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

# Only prod deps for runtime
COPY package*.json ./
RUN npm ci --omit=dev

# Copy build artifacts and public assets
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

EXPOSE 3000
CMD ["npm", "run", "start"]