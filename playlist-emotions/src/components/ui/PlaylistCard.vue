<script setup lang="ts">
import type { ClusterPlaylistBody, ClusterPlaylistResponse, Playlist } from '@/types/PlayList.ts'
import SpinnerComponent from '@/components/ui/SpinnerComponent.vue'
import { ref } from 'vue'

const props = defineProps<Playlist>()
const loading = ref<boolean>(false)

const api = import.meta.env.VITE_API_URL

async function createClusterPlaylist(): Promise<ClusterPlaylistResponse> {
  loading.value = true

  // Optional: abort if server hangs
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort('timeout'), 15000)

  try {
    const body: ClusterPlaylistBody = {
      description: props.description,
      public: false,
      replace: true,
    }

    const res = await fetch(`${api}/clusters/${props.cluster_id}/playlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    if (!res.ok) {
      // Best-effort JSON parse for error payloads
      let detail = res.statusText
      try {
        const err = await res.json()
        detail = err?.detail || detail
      } catch {
        /* ignore parse error */
      }
      throw new Error(`Failed to create playlist (${res.status}): ${detail}`)
    }

    // Success path
    const response = (await res.json()) as ClusterPlaylistResponse

    // Try app, fallback to web
    openInSpotify(response, true)

    return response
  } catch (err) {
    // Bubble up so caller can toast/log
    throw err
  } finally {
    // Always release UI and timers
    clearTimeout(timeoutId)
    loading.value = false
  }
}

function openInSpotify(resp: ClusterPlaylistResponse, preferApp = true): void {
  const webUrl = resp.playlist_url
  if (!preferApp) {
    window.open(webUrl, '_blank', 'noopener,noreferrer')
    return
  }

  const appUrl = `spotify:playlist:${resp.playlist_id}`

  // Try app first, then fallback to web after a short delay
  const fallback = setTimeout(() => {
    window.open(webUrl, '_blank', 'noopener,noreferrer')
  }, 700)

  // iOS/Safari behave better with location changes than window.open
  try {
    window.location.href = appUrl
  } catch {
    clearTimeout(fallback)
    window.open(webUrl, '_blank', 'noopener,noreferrer')
  }
}
</script>

<template>
  <SpinnerComponent :active="loading" title="Creating Playlist" />
  <div class="rounded-2xl glass shadow-lg w-full h-[600px] flex flex-col justify-between">
    <div class="p-5">
      <h2 class="text-xl font-bold text-white">{{ title }}</h2>
      <p class="mt-2 text-sm text-white/70">{{ description }}</p>
      <div class="flex flex-row justify-between items-center gap-2 mt-2 text-xs text-white/50">
        <p>Total Tracks: {{ tracks.length }}</p>
        <p>|</p>
        <button
          class="underline hover:text-white cursor-pointer transition-all duration-200"
          @click="createClusterPlaylist"
        >
          Create or Update Playlist
        </button>
      </div>
    </div>

    <ul class="mt-4 space-y-2 px-2 h-full overflow-y-auto">
      <li
        v-for="(t, i) in tracks"
        :key="t.cluster_id ?? i"
        class="flex items-center gap-3 rounded-lg px-3 py-2 text-white/80 hover:bg-white/5"
      >
        <span class="text-sm text-white/50">{{ i + 1 }}.</span>
        <div class="min-w-0">
          <p class="truncate font-medium text-white">{{ t.track_name }}</p>
          <p class="truncate text-xs text-white/60">
            {{ t.artist_name }} <span v-if="t.album_name" class="text-white/40">•</span>
            {{ t.album_name }}
          </p>
        </div>
      </li>
    </ul>
  </div>
</template>
