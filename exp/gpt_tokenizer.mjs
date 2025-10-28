const {
  encode,
  encodeChat,
  decode,
  isWithinTokenLimit,
  encodeGenerator,
  decodeGenerator,
  decodeAsyncGenerator,
  ALL_SPECIAL_TOKENS,
} = require('gpt-tokenizer')


function tokenize(text){
  let tokens = encode(text)
  return tokens
}
