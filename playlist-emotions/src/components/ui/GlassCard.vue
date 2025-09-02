<!-- components/TechBorderCard.vue -->
<template>
  <div
    class="relative rounded-2xl border-2 border-emerald-400/80
           p-6 bg-transparent glass"
    :class="hover ? 'transition-transform duration-300 hover:-translate-y-0.5 hover:shadow-[0_0_20px_2px_rgba(16,185,129,0.6)]' : ''"
    v-bind="$attrs"
  >

    <div class="relative flex items-start gap-4">
      <slot name="icon" />
      <div class="w-full">
        <h3 v-if="title" class="text-emerald-300 text-lg font-semibold tracking-tight text-center w-full mb-5">
          {{ title }}
        </h3>
        <p v-if="subtitle" class="text-emerald-200/70 text-sm mt-1">
          {{ subtitle }}
        </p>
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
