<script setup lang="ts">
import { defineEmits, onMounted } from 'vue';
import katex from 'katex';

const fonts = ['+', '-', '\\times', '\\div', '\\times', '\\cap'];
const arrows = ['\\rightarrow', '\\leftarrow', '\\uparrow', '\\downarrow'];

const emit = defineEmits(['insert']);

const emitSymbol = (symbol: string) => {
    emit('insert', symbol);
};

const renderKatex = () => {
    const elements = document.querySelectorAll('.katex-symbol');
    elements.forEach((el) => {
        katex.render(el.textContent || '', el as HTMLElement);
    });
};

onMounted(() => {
    renderKatex();
});
</script>

<template>
    <div class="editor">
        <el-dropdown trigger="hover">
            <span class="dropdown-link">
                字体
            </span>
            <template #dropdown>
                <el-row class="editor-panel">
                    <button v-for="symbol in fonts" :key="symbol" class="button" @click="emitSymbol(symbol)">
                        <span class="katex-symbol">{{ symbol }}</span>
                    </button>
                </el-row>
            </template>
        </el-dropdown>
    </div>
</template>

<style scoped>
.editor {
    display: flex;
    justify-content: center;
}

.editor-panel {
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.dropdown-link {
    cursor: pointer;
    border: 1px solid #409eff;
    border-radius: 3px;
    padding: 5px;
    color: #409eff;
    transition: 0.3s ease-in-out;
}

.dropdown-link:hover {
    background-color: #409eff;
    color: white;
}

.button {
    cursor: pointer;
    border: 1px solid #409eff;
    border-radius: 3px;
    padding: 5px;
    color: #409eff;
    transition: 0.3s ease-in-out;
    background-color: white;
}

.button:hover {
    background-color: #409eff;
    color: white;
}
</style>
