<script setup lang="ts">
import { defineEmits, onMounted } from 'vue';
import katex from 'katex';

const binaryOperations = ['+', '-', '*', '\\div', '\\times', '\\cap'];
const arrows = ['\\rightarrow', '\\leftarrow', '\\uparrow', '\\downarrow'];

const emit = defineEmits(['insert']);

const emitSymbol = (symbol: string) => {
    emit('insert', symbol); // 向父组件传递符号
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
        <span class="el-dropdown-link">
            快捷键
        </span>
        <template #dropdown>
            <el-row class="shortcut-panel">
                <el-col :span="24">
                    <div class="shortcut-category">二元运算符 Binary operations</div>
                    <div class="shortcut-symbols">
                        <el-button v-for="symbol in binaryOperations" :key="symbol" size="mini"
                            @click="emitSymbol(symbol)">
                            <span class="katex-symbol">{{ symbol }}</span>
                        </el-button>
                    </div>
                </el-col>
                <el-col :span="24">
                    <div class="shortcut-category">箭头符号 Arrows</div>
                    <div class="shortcut-symbols">
                        <el-button v-for="symbol in arrows" :key="symbol" size="mini" @click="emitSymbol(symbol)">
                            <span class="katex-symbol">{{ symbol }}</span>
                        </el-button>
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
</style>
