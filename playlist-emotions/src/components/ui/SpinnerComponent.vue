<script setup lang="ts">
// Comments in English only.

import { watch, onUnmounted, withDefaults, defineProps } from 'vue'

const props = withDefaults(
  defineProps<{
    active: boolean
    title?: string
    message?: string
  }>(),
  {
    title: 'Creating Playlist…',
    message: 'Please wait.',
  },
)

// Lock/unlock document scroll when 'active' changes
const html = typeof document !== 'undefined' ? document.documentElement : null
const unlock = () => {
  if (html) html.style.overflow = ''
}

watch(
  () => props.active,
  (val) => {
    if (html) html.style.overflow = val ? 'hidden' : ''
  },
  { immediate: true },
)

onUnmounted(unlock)
</script>

<template>
  <transition name="fade">
    <div
      v-if="active"
      class="fixed inset-0 z-[1000] flex items-center justify-center bg-neutral-950/70 backdrop-blur-sm"
      aria-busy="true"
      aria-live="polite"
      role="alert"
    >
      <div
        class="rounded-2xl bg-neutral-900 px-8 py-6 ring-1 ring-white/10 shadow-2xl max-w-sm w-full text-center"
      >
        <div
          class="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-4 border-white/20 border-t-white"
        ></div>
        <h3 class="text-white text-lg font-semibold">{{ title }}</h3>
        <p class="mt-1 text-white/70 text-sm">{{ message }}</p>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
