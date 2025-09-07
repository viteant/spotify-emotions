<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { Icon } from '@iconify/vue'
import type { Filters } from '@/types/TracksEmotions.ts'

type Placement = 'left' | 'right'

const props = defineProps<{
  initial?: Partial<Filters>
  placement?: Placement
  loading?: boolean
  className?: string
}>()

const emit = defineEmits<{
  (e: 'apply', payload: Filters): void
  (e: 'clear'): void
}>()

const open = ref(false)
const artist = ref(props.initial?.artist ?? '')
const track = ref(props.initial?.track ?? '')
const album = ref(props.initial?.album ?? '')

const placement = props.placement ?? 'right'
const loading = props.loading ?? false
const className = props.className ?? ''

const btnRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)

function toggle() {
  open.value = !open.value
}

function handleClear() {
  artist.value = ''
  track.value = ''
  album.value = ''
  emit('apply', {
    artist: '',
    album: '',
    track: '',
  })
  open.value = false
}

function handleApply() {
  emit('apply', {
    artist: artist.value.trim(),
    track: track.value.trim(),
    album: album.value.trim(),
  })
  open.value = false
}

function onClickOutside(e: MouseEvent) {
  if (!open.value) return
  const t = e.target as Node
  if (panelRef.value?.contains(t) || btnRef.value?.contains(t)) return
  open.value = false
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') open.value = false
}

onMounted(() => {
  window.addEventListener('mousedown', onClickOutside)
  window.addEventListener('keydown', onKey)
})

onBeforeUnmount(() => {
  window.removeEventListener('mousedown', onClickOutside)
  window.removeEventListener('keydown', onKey)
})
</script>

<template>
  <div class="relative inline-block" :class="className">
    <button
      ref="btnRef"
      type="button"
      aria-haspopup="true"
      :aria-expanded="open ? 'true' : 'false'"
      @click="toggle"
      class="inline-flex items-center gap-2 rounded-2xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
    >
      <Icon icon="solar:filter-linear" class="h-5 w-5" />
      <span>Filters</span>
      <Icon :icon="open ? 'mdi:chevron-up' : 'mdi:chevron-down'" class="h-4 w-4 opacity-70" />
    </button>

    <!-- Dropdown panel -->
    <div
      ref="panelRef"
      role="dialog"
      aria-label="Panel de filtros"
      :class="[
        'absolute z-50 mt-2 w-[22rem] origin-top rounded-2xl border border-slate-200 bg-white p-4 shadow-xl transition-all duration-150',
        placement === 'right' ? 'right-0' : 'left-0',
        open ? 'scale-100 opacity-100' : 'pointer-events-none scale-95 opacity-0',
      ]"
    >
      <div class="flex items-center justify-between pb-2">
        <div class="flex items-center gap-2 text-slate-700">
          <Icon icon="ph:funnel" class="h-5 w-5" />
          <h3 class="text-sm font-semibold">Filters</h3>
        </div>
        <button
          type="button"
          @click="handleClear"
          class="inline-flex items-center gap-1 rounded-xl border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
        >
          <Icon icon="mdi:broom" class="h-4 w-4" />
          Clean
        </button>
      </div>

      <div class="space-y-3">
        <!-- Field: Artist -->
        <label class="block text-sm">
          <span class="mb-1 flex items-center gap-2 text-slate-600">
            <Icon icon="mdi:account-music" class="h-4 w-4" />
            Artist
          </span>
          <input
            v-model="artist"
            placeholder="Artist Name"
            class="text-black w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500"
          />
        </label>

        <!-- Field: Track -->
        <label class="block text-sm">
          <span class="mb-1 flex items-center gap-2 text-slate-600">
            <Icon icon="mdi:music-note" class="h-4 w-4" />
            Track
          </span>
          <input
            v-model="track"
            placeholder="Track Title"
            class="text-black w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500"
          />
        </label>

        <!-- Field: Album -->
        <label class="block text-sm">
          <span class="mb-1 flex items-center gap-2 text-slate-600">
            <Icon icon="mdi:album" class="h-4 w-4" />
            Album
          </span>
          <input
            v-model="album"
            placeholder="Album Name"
            class="text-black w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500"
          />
        </label>
      </div>

      <div class="mt-4 flex items-center justify-end gap-2">
        <button
          type="button"
          @click="open = false"
          class="rounded-xl border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Cancel
        </button>
        <button
          type="button"
          :disabled="loading"
          @click="handleApply"
          class="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <Icon icon="mdi:magnify" class="h-4 w-4" />
          {{ loading ? 'Filtering...' : 'Filter' }}
        </button>
      </div>
    </div>
  </div>
</template>
