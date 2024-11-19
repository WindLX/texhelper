<script setup lang="ts">
import { defineEmits, onMounted } from 'vue';
import katex from 'katex';

const binaryOperations = ['+', '-', '\\times', '\\div', '\\times', '\\cap'];
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
    <el-dropdown trigger="hover">
        <span class="dropdown-link">
            常用符号
        </span>
        <template #dropdown>
            <el-row class="shortcut-panel">
                <el-col :span="24">
                    <div class="shortcut-category">二元运算符 Binary operations</div>
                    <div class="shortcut-symbols">
                        <button v-for="symbol in binaryOperations" :key="symbol" class="button"
                            @click="emitSymbol(symbol)">
                            <span class="katex-symbol">{{ symbol }}</span>
                        </button>
                    </div>
                </el-col>
                <el-col :span="24">
                    <div class="shortcut-category">箭头符号 Arrows</div>
                    <div class="shortcut-symbols">
                        <button v-for="symbol in arrows" :key="symbol" class="button" @click="emitSymbol(symbol)">
                            <span class="katex-symbol">{{ symbol }}</span>
                        </button>
                    </div>
                </el-col>
            </el-row>
        </template>
    </el-dropdown>
</template>

<style scoped>
.shortcut-panel {
    padding: 10px;
}

.shortcut-category {
    font-weight: bold;
    margin-top: 5px;
    margin-bottom: 5px;
}

.shortcut-symbols {
    display: flex;
    flex-wrap: wrap;
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
