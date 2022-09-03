use std::io::ErrorKind;

pub fn retrieve<'a, T>(input: &'a Vec<T>, index: usize) -> Result<&'a T, ErrorKind> {
    let option = input.get(index);
    match option {
        Some(x) => Ok(x),
        _ => Err(ErrorKind::NotFound)
    }
}

pub fn retrieve_mut<'a, T>(input: &'a mut Vec<T>, index: usize) -> Result<&'a mut T, ErrorKind> {
    let option = input.get_mut(index);
    match option {
        Some(x) => Ok(x),
        _ => Err(ErrorKind::NotFound)
    }
}