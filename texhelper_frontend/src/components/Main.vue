<script setup lang="ts">
import { ref } from 'vue'
import katex from 'katex'
import temml from 'temml'
import { ElNotification } from 'element-plus'
import ShortcutMenu from './ShortcutMenu.vue'

const latexCode = ref('')
const renderedLatex = ref('')

const image = ref<File | null>(null)
const imageUrl = ref('')
const tools = ['x', 'αβγ', 'x^y', '√x', 'lim', 'sin', '∫', '∑', '{[]}'];

const isLoading = ref(false)
const uploadFileButton = ref<HTMLInputElement | null>(null)

const triggerUpload = () => {
  uploadFileButton.value?.click()
}

const handleImageUpload = (event: Event) => {
  const target = event.target as HTMLInputElement
  const uploadFile = target.files?.[0]
  if (uploadFile) {
    image.value = uploadFile
    imageUrl.value = URL.createObjectURL(uploadFile)
    imageOcr().then(text => {
      latexCode.value = text
      renderLatex()
    }).catch(err => {
      ElNotification({
        title: 'Error',
        message: err,
        type: 'error',
      })
    })
  }
}

const handelImagePaste = (event: ClipboardEvent) => {
  const items = event.clipboardData?.items
  if (!items) {
    return
  }
  for (const item of items) {
    if (item.type.indexOf('image') === -1) {
      continue
    }
    const file = item.getAsFile()
    if (!file) {
      continue
    }
    image.value = file
    imageUrl.value = URL.createObjectURL(file)
    imageOcr().then(text => {
      latexCode.value = text
      renderLatex()
    }).catch(err => {
      ElNotification({
        title: 'Error',
        message: err,
        type: 'error',
      })
    })
  }
}

const imageOcr: () => Promise<string> = async () => {
  if (!image.value) {
    throw new Error('No image uploaded')
  }
  const formData = new FormData()
  formData.append('image', image.value)
  isLoading.value = true
  try {
    const response = await fetch('http://localhost:8000/ocr', {
      method: 'POST',
      body: formData
    })
    if (!response.ok) {
      throw new Error('Failed to OCR image')
    }
    const data = await response.json()
    return data.text
  } finally {
    isLoading.value = false
  }
}

const renderLatex = () => {
  renderedLatex.value = katex.renderToString(latexCode.value, { displayMode: true })
}

const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text).then(() => {
    ElNotification({
      title: 'Success',
      message: 'Copied to clipboard',
      type: 'success',
    })
  }).catch(err => {
    ElNotification({
      title: 'Error',
      message: 'Could not copy text: ' + err,
      type: 'error',
    })
  })
}

const copyLatexToClipboard = () => {
  copyToClipboard(latexCode.value)
}

const copyMathMLToClipboard = () => {
  const mathML = temml.renderToString(latexCode.value)
  copyToClipboard(mathML)
}

</script>

<template>
  <div>
    <el-container>
      <el-header>LaTeX 编辑器</el-header>
      <el-main v-loading="isLoading">
        <el-row>
          <el-card class="card">
            <h3>输入区域 Input</h3>
            <el-tabs class="tabs">
              <el-tab-pane label="快捷工具">
                <ShortcutMenu />
                <el-button-group>
                  <el-button v-for="tool in tools" :key="tool" size="small">{{ tool }}</el-button>
                </el-button-group>
              </el-tab-pane>
              <el-tab-pane label="公式模板">
                <el-button-group>
                  <el-button v-for="tool in tools" :key="tool" size="small">{{ tool }}</el-button>
                </el-button-group>
              </el-tab-pane>
              <el-tab-pane label="OCR">
                <div class="image" @paste="handelImagePaste">
                  <div class="upload" @click="triggerUpload">
                    <div class="content">上传图片 或 在此处粘贴</div>
                    <input type="file" @change="handleImageUpload" class="input-button" ref="uploadFileButton" />
                  </div>
                  <el-image v-if="imageUrl" :src="imageUrl"></el-image>
                  <el-empty v-else></el-empty>
                </div>
              </el-tab-pane>
            </el-tabs>
            <el-input v-model="latexCode" @input="renderLatex" type="textarea" :rows="10" class="latex-input"
              placeholder="请输入LaTeX表达式"></el-input>
          </el-card>
        </el-row>
        <el-row>
          <el-card class="card">
            <h3>输出区域 Output</h3>
            <div v-html="renderedLatex" class="latex-output"></div>
            <el-button-group>
              <el-button @click="copyLatexToClipboard">复制Latex</el-button>
              <el-button @click="copyMathMLToClipboard">复制MathML</el-button>
            </el-button-group>
          </el-card>
        </el-row>
      </el-main>
    </el-container>
  </div>
</template>

<style>
.tabs {
  margin-bottom: 10px;
}

.el-textarea__inner {
  font-family: 'Source Code Pro', 'Hack Nerd Font', 'Courier New', Courier, monospace;
}

.latex-output {
  font-size: 1.5em;
  border: 1px solid #ddd;
  padding: 10px;
  margin-top: 10px;
  margin-bottom: 10px;
  min-height: 200px;
}

.el-row {
  margin-bottom: 20px;
}

.el-row:last-child {
  margin-bottom: 0;
}

.card {
  width: 100%;
}

.upload {
  display: block;
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  background-color: #f5f7fa;
  cursor: pointer;
  text-align: center;
  color: #606266;
  font-size: 14px;
  transition: background-color 0.3s;

  &:hover {
    background-color: #e6e8eb;
  }

  &:focus {
    outline: none;
    border-color: #409eff;
  }
}

.upload .input-button {
  display: none;
}
</style>
