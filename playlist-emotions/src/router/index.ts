import { createRouter, createWebHistory } from 'vue-router'
import IndexPage from '../page/IndexPage.vue'
import PlaylistPage from '@/page/PlaylistPage.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', component: IndexPage },
    { path: '/suggested-playlists', component: PlaylistPage },
  ],
})

export default router
