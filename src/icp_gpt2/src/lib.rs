//use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
//use ic_stable_structures::{DefaultMemoryImpl, StableBTreeMap}; //, Storable};
use ic_stable_structures::{memory_manager::{MemoryId, MemoryManager}, DefaultMemoryImpl};
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;
use candid::{CandidType, Deserialize};
use crate::onnx::MEMORY_MANAGER;
use std::cell::RefCell;
use crate::onnx::{model_inference};

//type Memory = VirtualMemory<DefaultMemoryImpl>;
mod onnx;
mod storage;

thread_local! {
    pub static TOKENIZER_STATE: RefCell<Option<Tokenizer>> = RefCell::new(None);
}

#[ic_cdk::update]
fn load_tokenizer() -> std::result::Result<String, String> {
    let bpe = BPE::from_file("vocab.json", "merges.txt")
        .build()
        .unwrap();
    
    let mut tokenizer = Tokenizer::new(bpe);
    
    // Store tokenizer
    TOKENIZER_STATE.with(|t| {
        *t.borrow_mut() = Some(tokenizer);
    });
    Ok("Tokenizer loaded successfully".to_string())
}

#[ic_cdk::query]
fn talk_with_agent(input: String) -> std::result::Result<String, String> {
    TOKENIZER_STATE.with(|t| {
        if let Some(ref tokenizer) = *t.borrow() {
            let encoding = tokenizer.encode(input, false)
                .map_err(|e| e.to_string())?;
            
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

            let result: Vec<i64> = model_inference(14, ids)
                .map_err(|e| e.to_string())?;
            
                let tokens: Vec<String> = result.iter()
                .map(|&id| tokenizer.decode(&[id as u32], false).unwrap())
                .collect();            
            Ok(tokens.join(" "))
        } else {
            Err("Tokenizer not loaded".to_string())
        }
    })
}

#[ic_cdk::init]
fn init() {
    // Initialize the WASI memory
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Initialize the application memory (StableBTreeMap)
    //let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    //MAP.with(|map| {
    //    *map.borrow_mut() = StableBTreeMap::init(app_memory);
    //});
}

#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    // Save any necessary state before upgrade if needed
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    // Reinitialize the WASI memory after upgrade
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    
    // Reinitialize the application memory (StableBTreeMap) after upgrade
    //let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    //MAP.with(|map| {
    //    *map.borrow_mut() = StableBTreeMap::init(app_memory);
    //});
}


//////////////////////////////////////////////////////////////////////



const MODEL_FILE: &str = "onnx_model.onnx";

/// Clears the face detection model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn clear_model_bytes() {
    storage::clear_bytes(MODEL_FILE);
}

/// Appends the given chunk to the face detection model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn append_model_bytes(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}

#[ic_cdk::update]
fn append_vocab_bytes(bytes: Vec<u8>) {
    storage::append_bytes("vocab.json", bytes);
}

#[ic_cdk::update]
fn append_merges_bytes(bytes: Vec<u8>) {
    storage::append_bytes("merges.txt", bytes);
}

#[ic_cdk::update]
fn clear_vocab_bytes() {
    storage::clear_bytes("vocab.json");
}

#[ic_cdk::update]
fn clear_merges_bytes() {
    storage::clear_bytes("merges.txt");
}

/// Returns the length of the model bytes.
#[ic_cdk::query]
fn model_bytes_length() -> usize {
    storage::bytes_length(MODEL_FILE)
}
