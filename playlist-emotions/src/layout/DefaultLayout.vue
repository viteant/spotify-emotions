<script setup lang="ts">
import { useRoute } from 'vue-router'
const $route = useRoute()
const routes = [
  { src: '/', title: 'Emotions' },
  { src: '/suggested-playlists', title: 'Suggested Playlists' },
]
</script>

<template>
  <header class="bg-gray-800 dark:bg-gray-900">
    <nav>
      <div class="container mx-auto py-5">
        <div class="flex justify-end text-lg">
          <RouterLink
            v-for="route in routes"
            :key="route.title"
            :to="route.src"
            class="mr-5 hover:text-emerald-300 transition duration-300"
            :class="{ 'text-emerald-300': $route.path == route.src }"
          >
            {{ route.title }}
          </RouterLink>
        </div>
      </div>
    </nav>
  </header>
  <div class="container py-5 mx-auto">
    <slot />
  </div>
</template>

<style scoped></style>
