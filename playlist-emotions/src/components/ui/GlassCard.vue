<!-- components/TechBorderCard.vue -->
<template>
  <div
    class="relative rounded-2xl border-2 border-emerald-400/80
           p-6 bg-transparent glass h-full"
    :class="hover ? 'transition-transform duration-300 hover:-translate-y-0.5 hover:shadow-[0_0_20px_2px_rgba(16,185,129,0.6)]' : ''"
    v-bind="$attrs"
  >

    <div class="relative flex flex-col items-center gap-4">
      <div class="w-full">
        <h3 v-if="title" class="text-emerald-300 text-lg font-semibold tracking-tight text-center w-full">
          {{ title }}
        </h3>
      </div>
      <div class="ml-auto">
        <slot name="actions" />
      </div>
    </div>

    <div class="relative mt-4 text-emerald-100/90">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string
  subtitle?: string
  hover?: boolean
}>()
</script>
